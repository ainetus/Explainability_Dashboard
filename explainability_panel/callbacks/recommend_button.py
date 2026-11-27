########################################################################################################
#                                           Recommend button callbacks
########################################################################################################
from dash import callback, Output, Input, dcc, no_update
from services import (my_agent, my_env, run_recommendation, explain_recommendation,
                      recommendation_highlights, episode_over_text)
from callbacks.common import stat_outputs, empty_stats, format_stats, RUNNING_BUTTONS


@callback(
    Output("action-text", "children", allow_duplicate=True),
    Output("expert-text", "children", allow_duplicate=True),
    Output("action-taken", "data"),
    Output("action-stats", "data"),
    Output("graphic-output", "children", allow_duplicate=True),
    Output("time-stamp", "children", allow_duplicate=True),
    Output("explanation-text", "children", allow_duplicate=True),
    *stat_outputs(),
    Input("action-button", "n_clicks"),
    running=RUNNING_BUTTONS,
    prevent_initial_call=True
)
def make_action(n):
    nothing = [no_update] * (7 + len(empty_stats()))
    if not n or not my_env.env_reset:
        return nothing
    if my_env.done:
        nothing[6] = episode_over_text(my_env.episode_status)
        return nothing
    if my_agent.agent is None:
        nothing[6] = "No agent is loaded: choose one and press Load Env.\n"
        return nothing

    # Expert's and agent's actions on the current state, then apply the agent's action
    stats = run_recommendation(my_env, my_agent)

    figure = no_update
    if not my_env.done:
        figure = dcc.Graph(figure=my_env.plot_obs(recommendation_highlights(stats)))
    return (my_agent.action.__str__(),
            my_env.expert_action.__str__(),
            True,
            stats,
            figure,
            my_env.get_time_stamp(),
            explain_recommendation(stats, my_env),
            *format_stats(stats))
