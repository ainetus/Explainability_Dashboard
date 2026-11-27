########################################################################################################
#                                       Load button callbacks
########################################################################################################
from dash import callback, Output, Input, State, dcc, no_update
from services import my_agent, my_env, describe_grid_state, AGENT_LABELS
from callbacks.common import (stat_outputs, empty_stats, RUNNING_BUTTONS,
                              AGENT_ACTION_PLACEHOLDER, EXPERT_ACTION_PLACEHOLDER)


@callback(
    Output("graphic-output", "children", allow_duplicate=True),
    Output("env-loaded", "data"),
    Output("explanation-text", "children", allow_duplicate=True),
    Output("env-name-output", "children", allow_duplicate=True),
    Output("time-stamp", "children", allow_duplicate=True),
    Output("action-text", "children", allow_duplicate=True),
    Output("expert-text", "children", allow_duplicate=True),
    *stat_outputs(),
    Input("load-button", "n_clicks"),
    State("agent-choice", "value"),
    running=RUNNING_BUTTONS,
    prevent_initial_call=True
)
def load_output(n, agent_kind):
    if not n:
        return [no_update] * (7 + len(empty_stats()))
    env, obs = my_env.reset()
    try:
        my_agent.load(env, agent_kind)
        text = describe_grid_state(obs, env) + f"Agent loaded: {my_agent.label}.\n"
        loaded = True
    except Exception as exc:
        text = f"Could not load the {AGENT_LABELS.get(agent_kind, agent_kind)} agent:\n{exc}\n"
        loaded = False
    return (dcc.Graph(figure=my_env.plot_overload_graph()),
            loaded,
            text,
            env.env_name,
            my_env.get_time_stamp(),
            AGENT_ACTION_PLACEHOLDER,
            EXPERT_ACTION_PLACEHOLDER,
            *empty_stats())
