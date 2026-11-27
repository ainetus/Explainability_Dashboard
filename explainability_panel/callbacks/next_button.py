########################################################################################################
#                                       Next button callbacks
########################################################################################################
from dash import callback, Output, Input, dcc, no_update
from services import my_env, describe_grid_state, episode_over_text, overloaded_substations
from callbacks.common import (stat_outputs, empty_stats, RUNNING_BUTTONS,
                              AGENT_ACTION_PLACEHOLDER, EXPERT_ACTION_PLACEHOLDER)


@callback(
    Output("graphic-output", "children", allow_duplicate=True),
    Output("explanation-text", "children", allow_duplicate=True),
    Output("time-stamp", "children", allow_duplicate=True),
    Output("action-text", "children", allow_duplicate=True),
    Output("expert-text", "children", allow_duplicate=True),
    *stat_outputs(),
    Input("next-button", "n_clicks"),
    running=RUNNING_BUTTONS,
    prevent_initial_call=True
)
def load_next_obs(n):
    if not n or not my_env.env_reset:
        return [no_update] * (5 + len(empty_stats()))
    if my_env.done:
        return (no_update, episode_over_text(my_env.episode_status), *[no_update] * (3 + len(empty_stats())))

    # Fast-forward to the next overload
    status = my_env.next_obs_to_overload()
    if status != "running":
        figure = no_update
        text = episode_over_text(status)
    else:
        overloaded = overloaded_substations(my_env.obs, my_env.env)
        figure = dcc.Graph(figure=my_env.plot_obs({"Overloaded line ends": (overloaded, "#fd7e14", 26, 1.0)}))
        text = describe_grid_state(my_env.obs, my_env.env)
    return (figure,
            text,
            my_env.get_time_stamp(),
            AGENT_ACTION_PLACEHOLDER,
            EXPERT_ACTION_PLACEHOLDER,
            *empty_stats())
