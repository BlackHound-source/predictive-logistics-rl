"""State representation, discretization, and feasibility masking."""

import numpy as np
from config import ACTIONS, ACTION_IDS, STATE_BINS


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def make_state(inventory, demand, weather, route, capacity, trend, priority):
    demand = max(float(demand), 1.0)
    return np.array(
        [
            clip01(float(inventory) / demand),
            clip01(float(demand) / 25000.0),
            clip01(weather),
            clip01(route),
            clip01(capacity),
            clip01((float(trend) + 0.20) / 0.70),
            clip01(priority),
        ],
        dtype=np.float32,
    )


def discretize(state: np.ndarray) -> tuple:
    bins = np.floor(state * STATE_BINS).astype(int)
    return tuple(int(v) for v in np.clip(bins, 0, STATE_BINS - 1))


def feasible_actions(inventory, demand, route, capacity):
    coverage = inventory / max(demand, 1.0)
    allowed = list(range(len(ACTIONS)))

    if route >= 0.70 and ACTION_IDS["ALTERNATE ROUTE"] in allowed:
        allowed.remove(ACTION_IDS["ALTERNATE ROUTE"])

    if capacity >= 0.70 and ACTION_IDS["RESERVE CAPACITY"] in allowed:
        allowed.remove(ACTION_IDS["RESERVE CAPACITY"])

    if coverage >= 1.10:
        for action in ["FORWARD REPLENISHMENT", "PRIORITY REPLENISHMENT"]:
            aid = ACTION_IDS[action]
            if aid in allowed:
                allowed.remove(aid)
    elif coverage >= 0.95 and ACTION_IDS["PRIORITY REPLENISHMENT"] in allowed:
        allowed.remove(ACTION_IDS["PRIORITY REPLENISHMENT"])

    return allowed if allowed else [ACTION_IDS["NORMAL SUPPLY"]]
