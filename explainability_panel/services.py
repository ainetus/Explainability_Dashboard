import os
import io
import configparser
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import PIL
import cairosvg
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as sps

import grid2op
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
from grid2op.PlotGrid import PlotPlotly
from lightsim2grid import LightSimBackend

from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph
from alphaDeesp.expert_operator import expert_operator

from ExpertAgent import ASSETS, ExpertAgentHeuristic
from explainability_panel import get_package_root

# A line is overloaded when its flow reaches its thermal limit (rho >= 1)
RHO_OVERLOAD = 1.0
# Max rho given to an action whose simulation fails: game over, exception, illegal or ambiguous action
RHO_INVALID = 2.0
# Number of the agent's preferred actions compared with the greedy ranking
TOP_K = 20
# Significance level of the Kendall and Wilcoxon tests
ALPHA = 0.05
# The actions ranked by the greedy strategy: the action space of the PPO agent
GREEDY_ACTION_POOL = os.path.join(ASSETS, "whole_action_space.npz")

AGENT_HEURISTIC = "heuristic"
AGENT_PPO = "ppo"
AGENT_LABELS = {AGENT_HEURISTIC: "Expert heuristic", AGENT_PPO: "PPO (RL)"}
# The released PPO model is trained on this environment
PPO_ENV_NAME = "ai4realnet_small"
# Set this environment variable to load another PPO model (path without the .zip extension)
PPO_MODEL_ENV_VAR = "PPO_MODEL_PATH"
PPO_NET_ARCH = [800, 1000, 1000, 800]


########################################################################################################
#                                           Helpers
########################################################################################################
def overloaded_lines(obs) -> np.ndarray:
    return np.where(obs.rho >= RHO_OVERLOAD)[0]


def overloaded_substations(obs, env) -> list:
    lines = overloaded_lines(obs)
    subs = set(env.line_or_to_subid[lines].tolist()) | set(env.line_ex_to_subid[lines].tolist())
    return sorted(int(s) for s in subs)


def describe_grid_state(obs, env, past: bool = False) -> str:
    """Load level of the grid, its overloaded lines and the substations at their ends."""
    is_, are = ("was", "were") if past else ("is", "are")
    text = f"The Power Grid {is_} charged at {obs.rho.max():.2f}.\n"
    lines = overloaded_lines(obs)
    if len(lines):
        text += f"The power lines {[int(l) for l in lines]} {are} overloaded.\n"
        text += f"Substations {overloaded_substations(obs, env)} {are} concerned.\n"
    return text


def simulated_max_rho(obs, action) -> float:
    """Max rho after the action, simulated from obs.

    RHO_INVALID when the simulation ends the episode, raises an exception, or the action is
    illegal or ambiguous: grid2op then replaces it with "do nothing", which would hide the failure.
    """
    sim_obs, _, done, info = obs.simulate(action)
    if done or info.get("exception") or info.get("is_illegal") or info.get("is_ambiguous"):
        return RHO_INVALID
    return float(sim_obs.rho.max())


def _nonzero_elements(values, to_subid, no_change=0.0) -> set:
    values = np.asarray(values, dtype=float)
    changed = np.abs(values - no_change) > 1e-6
    return set(np.asarray(to_subid)[changed].tolist())


def action_substations(action, env, line_status=None) -> list:
    """Substations touched by an action.

    The topological part (bus and line status changes) is used when there is one: it is what can be
    compared with the expert, whose actions are topological. Otherwise, the substations of the
    generators and storage units whose injections are changed (redispatching, curtailment, storage).
    """
    lines, subs = action.get_topological_impact(line_status)
    topo = set(np.where(subs)[0].tolist())
    impacted_lines = np.where(lines)[0]
    topo |= set(env.line_or_to_subid[impacted_lines].tolist())
    topo |= set(env.line_ex_to_subid[impacted_lines].tolist())
    if topo:
        return sorted(int(s) for s in topo)

    injections = set()
    injections |= _nonzero_elements(action.redispatch, env.gen_to_subid)
    injections |= _nonzero_elements(action.curtail, env.gen_to_subid, no_change=-1.0)
    if env.n_storage:
        injections |= _nonzero_elements(action.storage_p, env.storage_to_subid)
    return sorted(int(s) for s in injections)


def substation_graph(env, line_status) -> nx.Graph:
    """Substations linked by the power lines in service."""
    graph = nx.Graph()
    graph.add_nodes_from(range(env.n_sub))
    for line_id in np.where(line_status)[0]:
        graph.add_edge(int(env.line_or_to_subid[line_id]), int(env.line_ex_to_subid[line_id]))
    return graph


def compute_distance(env, obs, agent_subs: list, expert_zone: list) -> Optional[int]:
    """Number of hops between the closest substation touched by the agent and the expert zone,
    on the grid as it was when the agent decided. None if one of them is empty or they are not connected."""
    if not agent_subs or not expert_zone:
        return None
    distances = nx.multi_source_dijkstra_path_length(substation_graph(env, obs.line_status), set(expert_zone))
    reachable = [distances[s] for s in agent_subs if s in distances]
    return int(min(reachable)) if reachable else None


def _float_or_none(value) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else float(value)


def compute_greedy_stats(obs, agent_action, pool: list, agent_top_k: Optional[list] = None) -> dict:
    """Compare the agent with the greedy strategy, which ranks every action of the pool by the max rho
    obtained when simulating it from obs (lower is better).

    For every agent:
      - greedy_rank: rank of the agent's action among the pool (1 is the best),
      - rho_regret: its simulated max rho minus the best one of the pool (negative: the agent did better).
    For an agent with an output distribution (agent_top_k: its k preferred actions, most preferred first):
      - Kendall's tau between the agent's preference order and the simulated max rho of these k actions.
        tau > 0: the agent prefers the actions the simulator finds better.
      - One-sided Wilcoxon signed-rank test between the simulated max rho of the agent's i-th preferred
        action and of the greedy i-th best action of the whole pool, i = 1..k. A p-value below ALPHA
        means the agent's preferred actions are significantly worse than the greedy ones.
    """
    pool_rho = np.array([simulated_max_rho(obs, action) for action in pool])
    agent_rho = simulated_max_rho(obs, agent_action)
    valid = pool_rho[pool_rho < RHO_INVALID]
    best_rho = float(valid.min()) if valid.size else None

    res = {
        "agent_sim_rho": agent_rho,
        "greedy_best_rho": best_rho,
        "greedy_rank": int(1 + np.sum(pool_rho < agent_rho)),
        "greedy_pool_size": len(pool),
        "rho_regret": None if best_rho is None else agent_rho - best_rho,
        "top_k": None,
        "kendall_tau": None,
        "kendall_pval": None,
        "wilcoxon_stat": None,
        "wilcoxon_pval": None,
    }
    if not agent_top_k:
        return res

    k = len(agent_top_k)
    agent_top_rho = np.array([simulated_max_rho(obs, action) for action in agent_top_k])
    greedy_top_rho = np.sort(pool_rho)[:k]
    res["top_k"] = k

    if np.ptp(agent_top_rho) > 0:
        tau = sps.kendalltau(np.arange(k), agent_top_rho)
        res["kendall_tau"] = _float_or_none(tau.statistic)
        res["kendall_pval"] = _float_or_none(tau.pvalue)

    if np.allclose(agent_top_rho, greedy_top_rho):
        # The agent's top-k is the greedy top-k, in the same order
        res["wilcoxon_stat"], res["wilcoxon_pval"] = 0.0, 1.0
    else:
        test = sps.wilcoxon(agent_top_rho, greedy_top_rho, alternative="greater")
        res["wilcoxon_stat"] = _float_or_none(test.statistic)
        res["wilcoxon_pval"] = _float_or_none(test.pvalue)
    return res


def load_greedy_pool(env) -> list:
    vects = np.load(GREEDY_ACTION_POOL, allow_pickle=True)["action_space"]
    return [env.action_space({})] + [env.action_space.from_vect(vect) for vect in vects]


########################################################################################################
#                                           Environment
########################################################################################################
class MyEnv():
    def __init__(self,
                 path_env: str,
                 expert_config_path: str = "config_expert.ini",
                 seed: int = 42,
                 chronic_id: int = 0):
        self.path_env = path_env
        self.env = None
        self.obs = None
        self.old_obs = None
        self.env_reset = False
        self.next = False
        self.done = False
        self.last_info = None
        self.config = configparser.ConfigParser()
        self.config.read(expert_config_path)
        self.seed = seed
        self.chronic_id = chronic_id
        self.old_rho = None
        self.current_rho = None
        self.expert_action = None
        self.expert_rho = None
        self.expert_zone = []
        self.greedy_pool = []

    def reset(self, seed=None, chronic_id=None):
        if self.env is not None:
            self.env.close()
        env = grid2op.make(dataset=self.path_env,
                           backend=LightSimBackend(),
                           reward_class=L2RPNSandBoxScore,
                           other_rewards={"reward": L2RPNReward})
        env.seed(self.seed if seed is None else seed)
        env.set_id(self.chronic_id if chronic_id is None else chronic_id)
        self.obs = env.reset()
        self.old_obs = None
        self.old_rho = float(self.obs.rho.max())
        self.current_rho = float(self.obs.rho.max())
        self.env = env
        self.env_reset = True
        self.done = False
        self.last_info = None
        self.expert_action = None
        self.expert_rho = None
        self.expert_zone = []
        self.greedy_pool = load_greedy_pool(env)
        return self.env, self.obs

    def _check_loaded(self):
        if not self.env_reset or self.env is None or self.obs is None:
            raise RuntimeError("The environment should be loaded at first")

    @property
    def episode_status(self) -> str:
        """running, game_over or end_of_scenario"""
        if not self.done:
            return "running"
        if self.last_info and self.last_info.get("exception"):
            return "game_over"
        return "end_of_scenario"

    def _step(self, action):
        self.obs, _, self.done, self.last_info = self.env.step(action)
        self.current_rho = float(self.obs.rho.max())

    def next_obs(self, action=None):
        self._check_loaded()
        if self.done:
            raise RuntimeError("The episode is over: load the environment again")
        if action is None:
            action = self.env.action_space({})
        self.old_obs = self.obs.copy()
        self.old_rho = float(self.obs.rho.max())
        self._step(action)
        self.next = True
        return self.obs

    def next_obs_to_overload(self) -> str:
        """Fast-forward with "do nothing" to the next time step with an overloaded line.

        Stops earlier if the episode ends. Returns the episode status.
        """
        self._check_loaded()
        if self.done:
            return self.episode_status
        self.old_obs = self.obs.copy()
        self.old_rho = float(self.obs.rho.max())
        do_nothing = self.env.action_space({})
        while True:
            self._step(do_nothing)
            if self.done or len(overloaded_lines(self.obs)):
                break
        self.next = True
        return self.episode_status

    def get_time_stamp(self):
        self._check_loaded()
        return self.env.time_stamp

    def _expert_simulation(self, ltc):
        return Grid2opSimulation(self.obs,
                                 self.env.action_space,
                                 self.env.observation_space,
                                 param_options=self.config["DEFAULT"],
                                 debug=False,
                                 ltc=ltc,
                                 plot=False)

    def suggest_expert_action(self):
        """Most efficient ExpertOp4Grid action whose simulation succeeds ("do nothing" if none).

        Also sets the expert zone: the substations ExpertOp4Grid identified and simulated.
        """
        self._check_loaded()
        do_nothing = self.env.action_space({})
        self.expert_zone = []
        self.expert_action = do_nothing
        self.expert_rho = simulated_max_rho(self.obs, do_nothing)

        ltc = [int(l) for l in overloaded_lines(self.obs)]
        if not ltc:
            return self.expert_action

        _, results, actions = expert_operator(self._expert_simulation(ltc), plot=False, debug=False)
        zone = pd.to_numeric(results["Substation ID"], errors="coerce").dropna().astype(int)
        self.expert_zone = sorted({int(s) for s in zone if 0 <= s < self.env.n_sub})

        efficacity = pd.to_numeric(results["Efficacity"], errors="coerce").fillna(-np.inf).to_numpy()
        for idx in np.argsort(-efficacity, kind="stable"):
            rho = simulated_max_rho(self.obs, actions[idx])
            if rho < RHO_INVALID:
                self.expert_action, self.expert_rho = actions[idx], rho
                break

        expert_subs = action_substations(self.expert_action, self.env, self.obs.line_status)
        self.expert_zone = sorted(set(self.expert_zone) | set(expert_subs))
        return self.expert_action

    def plot_obs(self, highlights: Optional[dict] = None):
        """Plot the current observation.

        highlights: {legend label: (list of substation ids, color, marker size, opacity)}
        """
        self._check_loaded()
        plot_helper = PlotPlotly(self.env.observation_space, width=700, height=500)
        fig = plot_helper.plot_obs(self.obs)
        if not highlights:
            return fig
        layout = plot_helper._grid_layout
        names = self.env.name_sub
        for label, (subs, color, size, opacity) in highlights.items():
            if not subs:
                continue
            fig.add_trace(go.Scatter(
                x=[layout[names[s]][0] for s in subs],
                y=[layout[names[s]][1] for s in subs],
                mode="markers",
                name=label,
                text=[f"substation {s}" for s in subs],
                hoverinfo="text+name",
                marker=dict(size=size, color="rgba(0,0,0,0)", opacity=opacity,
                            line=dict(width=4, color=color)),
            ))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))
        return fig

    def plot_overload_graph(self):
        self._check_loaded()
        ltc = [int(l) for l in overloaded_lines(self.obs)]
        if not ltc:
            return self.plot_obs()
        sim = self._expert_simulation(ltc)
        g_over = OverFlowGraph(sim.topo, ltc, sim.get_dataframe())
        rescale_factor = 1  # for better layout, you can play with it to change the zoom level
        layout_rescale = [(e[0] / rescale_factor, e[1] / rescale_factor) for e in sim.layout]
        svg = g_over.plot(layout_rescale, save_folder="")
        im = PIL.Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg,
                                                        dpi=500,
                                                        output_width=1920,
                                                        output_height=1080,
                                                        scale=5))).convert("RGBA")
        fig = go.Figure(px.imshow(im).data)
        fig.update_layout(title="Overload graph", coloraxis_showscale=False,
                          height=500, margin=dict(l=0, r=0, t=40, b=0))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig


########################################################################################################
#                                           Agent
########################################################################################################
def load_ppo_agent(env, model_path: Optional[str] = None, top_k: int = TOP_K):
    """Load the ExpertAgentRL (PPO) agent and the gym environment it observes and acts through.

    model_path: path to the model without its .zip extension. By default, the PPO_MODEL_PATH environment
    variable, else the released model of the ExpertAgent package (downloaded on first use).
    """
    from stable_baselines3.ppo import MlpPolicy
    from ExpertAgent import ExpertAgentRL
    from ExpertAgent.utils.helper_functions import make_gymenv

    if model_path is None:
        model_path = os.environ.get(PPO_MODEL_ENV_VAR)
    if model_path is None:
        try:
            from ExpertAgent.utils.local_data import get_release_model
        except ImportError:
            raise RuntimeError(f"This version of the ExpertAgent package has no released PPO model: set the "
                               f"{PPO_MODEL_ENV_VAR} environment variable to the path of a model "
                               f"(without its .zip extension)") from None
        model_path = get_release_model("PPO_SB3", PPO_ENV_NAME)

    gym_env = make_gymenv(env, obs_attr_to_keep=["rho"], action_space_path="read_from_file")
    # The network is built from nn_kwargs, then its weights are replaced by load. Giving the model path to
    # the constructor instead would load it before top_k is set.
    nn_kwargs = {
        "policy": MlpPolicy,
        "env": gym_env,
        "verbose": False,
        "policy_kwargs": {"net_arch": PPO_NET_ARCH},
        "device": "cpu",
    }
    agent = ExpertAgentRL(name="PPO_SB3",
                          env=env,
                          action_space=env.action_space,
                          gymenv=gym_env,
                          gym_act_space=gym_env.action_space,
                          gym_obs_space=gym_env.observation_space,
                          nn_kwargs=nn_kwargs,
                          top_k=top_k,
                          device="cpu")
    agent.load(model_path)
    return agent, gym_env


class MyAgent():
    def __init__(self):
        self.kind = None
        self.agent = None
        self.gym_env = None
        self.action = None

    @property
    def label(self) -> str:
        return AGENT_LABELS.get(self.kind, "No agent")

    @property
    def has_distribution(self) -> bool:
        """Whether the agent ranks its actions (needed by the Kendall and Wilcoxon tests)."""
        return self.kind == AGENT_PPO

    def load(self, env, kind: str = AGENT_HEURISTIC, model_path: Optional[str] = None):
        self.agent, self.gym_env, self.kind, self.action = None, None, None, None
        if kind == AGENT_HEURISTIC:
            self.agent = ExpertAgentHeuristic(action_space=env.action_space, env=env)
        elif kind == AGENT_PPO:
            self.agent, self.gym_env = load_ppo_agent(env, model_path)
        else:
            raise ValueError(f"Unknown agent {kind!r}")
        self.kind = kind

    def my_act(self, obs):
        if self.agent is None:
            raise RuntimeError("Agent not loaded")
        self.action = self.agent.act(obs, reward=None)
        return self.action

    def top_k_actions(self, obs, k: int = TOP_K) -> Optional[list]:
        """The agent's k preferred actions, most preferred first: the highest logits of the PPO policy.
        None for an agent without output distribution."""
        if not self.has_distribution:
            return None
        gym_obs = self.gym_env.observation_space.to_gym(obs)
        act_ids = self.agent.topo_search.get_top_k(gym_obs, top_k=k)
        return [self.gym_env.action_space.from_gym(int(i)) for i in act_ids]


########################################################################################################
#                                           Recommendation
########################################################################################################
def run_recommendation(my_env: MyEnv, my_agent: MyAgent) -> dict:
    """Get the expert's and the agent's actions, compare them by simulation, then apply the agent's action.

    Every simulation is done before the step: obs.simulate always starts from the current state of the
    environment. All the values are JSON serializable (stored in the dashboard).
    """
    my_env._check_loaded()
    if my_agent.agent is None:
        raise RuntimeError("Agent not loaded")
    env, obs = my_env.env, my_env.obs

    expert_action = my_env.suggest_expert_action()
    agent_action = my_agent.my_act(obs)
    do_nothing = env.action_space({})

    greedy = compute_greedy_stats(obs, agent_action, my_env.greedy_pool,
                                  my_agent.top_k_actions(obs, TOP_K))
    agent_subs = action_substations(agent_action, env, obs.line_status)
    expert_subs = action_substations(expert_action, env, obs.line_status)
    distance = compute_distance(env, obs, agent_subs, my_env.expert_zone)

    my_env.next_obs(agent_action)

    stats = {
        "agent": my_agent.kind,
        "agent_has_distribution": my_agent.has_distribution,
        "agent_did_nothing": agent_action == do_nothing,
        "expert_did_nothing": expert_action == do_nothing,
        "agent_subs": agent_subs,
        "expert_subs": expert_subs,
        "expert_zone": list(my_env.expert_zone),
        "distance": distance,
        # Measured on the real step; meaningless if the step ended the episode
        "rho_imp": None if my_env.done else my_env.old_rho - my_env.current_rho,
        # Both simulated from the same observation, so they are comparable: > 0 means the agent did worse
        "rho_diff": greedy["agent_sim_rho"] - my_env.expert_rho,
        "expert_sim_rho": my_env.expert_rho,
        "episode_status": my_env.episode_status,
    }
    stats.update(greedy)
    return stats


def episode_over_text(status: str) -> str:
    if status == "game_over":
        return "GAME OVER: the grid collapsed. Press Load Env to start again.\n"
    return "End of the scenario. Press Load Env to start again.\n"


def explain_recommendation(stats: dict, my_env: MyEnv) -> str:
    """Textual explanation of the statistics of a recommendation."""
    text = f"The Power Grid was charged at {my_env.old_rho:.2f} before applying the Agent's action.\n"
    text += describe_grid_state(my_env.old_obs, my_env.env, past=True).split("\n", 1)[1]

    if stats["episode_status"] != "running":
        text += episode_over_text(stats["episode_status"])

    distance = stats["distance"]
    if stats["agent_did_nothing"]:
        text += "The agent did not act.\n"
    elif not stats["expert_zone"]:
        text += "No line was overloaded: the expert has no zone of interest.\n"
    elif distance is None:
        text += "The action of the agent is not connected to the zone suggested by the expert.\n"
    elif distance <= 2:
        text += "The action selected by the agent is very close to the zone suggested by the expert.\n"
    elif distance <= 5:
        text += "The action selected by the agent is somewhat far from the zone suggested by the expert.\n"
    else:
        text += "The action is independent from the zone suggested by the expert. See the other statistics for its performance.\n"

    rho_imp = stats["rho_imp"]
    if rho_imp is not None:
        if abs(rho_imp) < 5e-3:
            text += f"The action recommended by the agent has no impact on the load of the grid, still charged at {my_env.current_rho:.2f}.\n"
        elif rho_imp > 0:
            text += f"The action recommended by the agent reduced the maximum load of the grid by {rho_imp:.2f}.\n"
        else:
            text += f"The action recommended by the agent increased the maximum load of the grid by {-rho_imp:.2f}.\n"

    expert = "the expert's suggestion to do nothing" if stats["expert_did_nothing"] else "the expert's action"
    rho_diff = stats["rho_diff"]
    if abs(rho_diff) < 5e-3:
        text += f"In simulation, the agent's action is as good as {expert}.\n"
    elif rho_diff > 0:
        text += f"In simulation, the agent's action leaves the grid {rho_diff:.2f} more charged than {expert}.\n"
    else:
        text += f"In simulation, the agent's action leaves the grid {-rho_diff:.2f} less charged than {expert}.\n"

    regret = stats["rho_regret"]
    if regret is not None:
        text += f"Among the {stats['greedy_pool_size']} topological actions ranked by a greedy simulation, the agent's action ranks {stats['greedy_rank']}"
        if regret < -5e-3:
            text += ": it beats them all (it may combine several types of actions).\n"
        elif regret < 5e-3:
            text += ", within 0.01 of the best one.\n"
        else:
            text += f"; the best one leaves the grid {regret:.2f} less charged.\n"

    if not stats["agent_has_distribution"]:
        text += f"The {AGENT_LABELS[stats['agent']]} agent does not rank its actions: the Kendall and Wilcoxon tests need the PPO agent.\n"
        return text

    k = stats["top_k"]
    tau, tau_p = stats["kendall_tau"], stats["kendall_pval"]
    if tau is None:
        text += f"The agent's {k} preferred actions all have the same simulated outcome: their order cannot be compared with the greedy ranking.\n"
    elif tau_p < ALPHA and tau > 0:
        text += f"The agent's preference order of its {k} preferred actions agrees with their simulated outcomes (Kendall tau {tau:.2f}).\n"
    elif tau_p < ALPHA:
        text += f"The agent prefers the actions that the simulation finds worse (Kendall tau {tau:.2f}): its ranking is not reliable here.\n"
    else:
        text += f"The agent's preference order is not significantly related to the simulated outcomes (Kendall tau {tau:.2f}).\n"

    if stats["wilcoxon_pval"] is not None:
        if stats["wilcoxon_pval"] < ALPHA:
            text += f"The agent's {k} preferred actions are significantly worse (95% confidence) than the {k} best greedy actions.\n"
        else:
            text += f"No significant difference (95% confidence) between the agent's {k} preferred actions and the {k} best greedy actions.\n"
    return text


def recommendation_highlights(stats: dict) -> dict:
    return {
        "Expert interest zone": (stats["expert_zone"], "rgba(220,53,69,0.35)", 34, 1.0),
        "Expert action": (stats["expert_subs"], "#dc3545", 26, 1.0),
        "Agent action": (stats["agent_subs"], "#007bff", 20, 1.0),
    }


ENV_PATH = os.path.join(get_package_root(), "ressources", "ai4realnet_small")
EXPERT_CONFIG_PATH = os.path.join(get_package_root(), "explainability_panel", "config_expert.ini")
my_env = MyEnv(path_env=ENV_PATH, expert_config_path=EXPERT_CONFIG_PATH)
my_agent = MyAgent()
