########################################################################################################
#                                   Helpers shared by the callbacks
########################################################################################################
from dash import Output

# The statistics boxes, in the order of format_stats
STAT_IDS = [
    "distance-output",
    "rho-imp-output",
    "rho-diff-output",
    "greedy-rank-output",
    "kendall-stat-output",
    "kendall-pval-output",
    "test-stat-output",
    "test-pval-output",
]

AGENT_ACTION_PLACEHOLDER = "Agent's action ... "
EXPERT_ACTION_PLACEHOLDER = "Expert action ... "

# Disable the three buttons while a callback runs
RUNNING_BUTTONS = [
    (Output("load-button", "disabled"), True, False),
    (Output("next-button", "disabled"), True, False),
    (Output("action-button", "disabled"), True, False),
]


def stat_outputs():
    return [Output(stat_id, "children", allow_duplicate=True) for stat_id in STAT_IDS]


def empty_stats():
    return [""] * len(STAT_IDS)


def _fmt(value, fmt, missing="n/a"):
    return missing if value is None else format(value, fmt)


def _fmt_pval(value):
    if value is not None and value < 1e-3:
        return "p-value < 0.001"
    return f"p-value = {_fmt(value, '.3f')}"


def format_stats(stats: dict) -> list:
    """The text of the statistics boxes, in the order of STAT_IDS."""
    if stats["agent_did_nothing"]:
        distance = "n/a (no action)"
    elif not stats["expert_zone"]:
        distance = "n/a (no overload)"
    elif stats["distance"] is None:
        distance = "not connected"
    else:
        distance = f"{stats['distance']} hop(s)"

    if stats["rho_regret"] is None:
        rank = "n/a"
    else:
        rank = f"{stats['greedy_rank']} / {stats['greedy_pool_size']} (regret {stats['rho_regret']:+.2f})"

    if stats["agent_has_distribution"]:
        tests = [
            f"τ = {_fmt(stats['kendall_tau'], '.2f')}",
            _fmt_pval(stats["kendall_pval"]),
            f"W = {_fmt(stats['wilcoxon_stat'], '.1f')}",
            _fmt_pval(stats["wilcoxon_pval"]),
        ]
    else:
        tests = ["n/a (PPO only)"] * 4

    return [
        distance,
        _fmt(stats["rho_imp"], ".2f"),
        f"{stats['rho_diff']:+.2f}",
        rank,
        *tests,
    ]
