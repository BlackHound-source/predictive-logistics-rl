# ============================================================
# TACTICAL PREDICTIVE LOGISTICS MANAGEMENT SYSTEM
# ============================================================
# RENDER-READY COMMAND CONSOLE EDITION
#
# CORE ENGINE
#   1. Multivariate Polynomial Demand Forecaster
#   2. Dynamic GIS Corridor Engine
#   3. TL-RL Agent
#   4. RD3P Controller
#   5. Feasibility Mask
#   6. 50/50 Normalized Consensus
#
# FRONTEND
#   Military Command Terminal / Field Operations Console
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import folium

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# PERFORMANCE FIX ONLY:
# Exact nearest-state lookup, replacing the slow Python
# min(..., key=...) search used repeatedly during RL training.
from scipy.spatial import cKDTree

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "Gradio is missing. Install dependencies with: "
        "pip install -r requirements.txt"
    ) from exc

warnings.filterwarnings("ignore")


# ============================================================
# 1. THEATER & NETWORK TOPOLOGY REGISTRY
# ============================================================

THEATERS = {
    "LADAKH_NORTHERN_AXIS": {
        "center": [34.4800, 77.6500],
        "zoom": 9,

        "nodes": {
            "DEPOT_MAIN": {
                "name": "Rear Staging Base (RSB-01)",
                "lat": 34.1526,
                "lon": 77.5771,
                "role": "Strategic Supply Depot",
            },

            "NODE_TRANSIT": {
                "name": "Intermediate Transit Chokepoint (ILN-North)",
                "lat": 34.3500,
                "lon": 77.6200,
                "role": "Transit Logistics Hub",
            },

            "FOB_ALPHA": {
                "name": "Forward Operating Base (FOB-Alpha)",
                "lat": 34.6000,
                "lon": 77.7000,
                "role": "Forward Echelon Base",
            },

            "POST_SIERRA": {
                "name": "Forward Terminal Post (Sierra)",
                "lat": 34.8500,
                "lon": 77.7800,
                "role": "Frontline Sector Node",
            },
        },

        "primary_route": [
            [34.1526, 77.5771],
            [34.3500, 77.6200],
            [34.6000, 77.7000],
            [34.8500, 77.7800],
        ],

        "alternate_route": [
            [34.1526, 77.5771],
            [34.2800, 77.4500],
            [34.5200, 77.5300],
            [34.8500, 77.7800],
        ],
    },

    "EASTERN_SECTOR_TAWANG": {
        "center": [27.5861, 91.8594],
        "zoom": 9,

        "nodes": {
            "DEPOT_MAIN": {
                "name": "Tezpur Forward Logistics Base",
                "lat": 26.6528,
                "lon": 92.7926,
                "role": "Strategic Supply Depot",
            },

            "NODE_TRANSIT": {
                "name": "Sela Pass Transit Chokepoint",
                "lat": 27.5050,
                "lon": 92.1030,
                "role": "Mountain Pass Transit",
            },

            "FOB_ALPHA": {
                "name": "Tawang Forward Operating Base",
                "lat": 27.5861,
                "lon": 91.8594,
                "role": "Forward Sector Depot",
            },

            "POST_SIERRA": {
                "name": "Bum La Forward Defensive Post",
                "lat": 27.7275,
                "lon": 91.8900,
                "role": "Tactical Frontier Post",
            },
        },

        "primary_route": [
            [26.6528, 92.7926],
            [27.5050, 92.1030],
            [27.5861, 91.8594],
            [27.7275, 91.8900],
        ],

        "alternate_route": [
            [26.6528, 92.7926],
            [27.3500, 92.4000],
            [27.6200, 91.9500],
            [27.7275, 91.8900],
        ],
    },
}


# ============================================================
# 2. DEMAND FORECASTING ENGINE
# ============================================================

def build_temporal_demand_model():
    """
    Trains a polynomial regression model on synthetic
    historical demand telemetry.
    """

    np.random.seed(42)

    t = np.arange(1, 37).reshape(-1, 1)

    base = (
        13500.0
        + 120.0 * t.flatten()
        + 1400.0 * np.sin(t.flatten() / 2.5)
        + np.random.normal(0, 250, 36)
    )

    model = make_pipeline(
        PolynomialFeatures(degree=2),
        Ridge(alpha=1.0),
    )

    model.fit(t, base)

    return model


demand_model = build_temporal_demand_model()


def forecast_demand(
    month_idx: int,
    operational_surge: float = 1.0,
    trend: float = 0.0,
) -> float:

    t_eval = np.array([[37 + month_idx]])

    raw_pred = demand_model.predict(t_eval)[0]

    trend_adjusted = raw_pred * (
        1.0 + float(trend) * 0.45
    )

    return float(
        max(
            1000.0,
            trend_adjusted * operational_surge,
        )
    )


# ============================================================
# 3. ACTION SPACE
# ============================================================

ACTIONS = [
    "NORMAL SUPPLY",
    "FORWARD REPLENISHMENT",
    "PRIORITY REPLENISHMENT",
    "ALTERNATE ROUTE",
    "RESERVE CAPACITY",
]

ACTION_IDS = {
    action: index
    for index, action in enumerate(ACTIONS)
}


STATE_BINS = 4
ALPHA = 0.12
GAMMA = 0.93


# ============================================================
# 4. STATE REPRESENTATION
# ============================================================

def clip01(value):
    return float(
        np.clip(
            float(value),
            0.0,
            1.0,
        )
    )


def make_state(
    inventory,
    demand,
    weather,
    route,
    capacity,
    trend,
    priority,
):

    demand = max(
        float(demand),
        1.0,
    )

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


def discretize(state):

    bins = np.floor(
        state * STATE_BINS
    ).astype(int)

    return tuple(
        int(v)
        for v in np.clip(
            bins,
            0,
            STATE_BINS - 1,
        )
    )


# ============================================================
# 5. FEASIBILITY ENGINE
# ============================================================

def feasible_actions(
    inventory,
    demand,
    route,
    capacity,
):

    coverage = inventory / max(
        demand,
        1.0,
    )

    allowed = list(
        range(len(ACTIONS))
    )

    if (
        route >= 0.70
        and ACTION_IDS["ALTERNATE ROUTE"] in allowed
    ):
        allowed.remove(
            ACTION_IDS["ALTERNATE ROUTE"]
        )

    if (
        capacity >= 0.70
        and ACTION_IDS["RESERVE CAPACITY"] in allowed
    ):
        allowed.remove(
            ACTION_IDS["RESERVE CAPACITY"]
        )

    if coverage >= 1.10:

        for action in [
            "FORWARD REPLENISHMENT",
            "PRIORITY REPLENISHMENT",
        ]:

            action_id = ACTION_IDS[action]

            if action_id in allowed:
                allowed.remove(action_id)

    elif (
        coverage >= 0.95
        and ACTION_IDS["PRIORITY REPLENISHMENT"] in allowed
    ):

        allowed.remove(
            ACTION_IDS["PRIORITY REPLENISHMENT"]
        )

    if not allowed:
        return [
            ACTION_IDS["NORMAL SUPPLY"]
        ]

    return allowed


# ============================================================
# 6. TL-RL AGENT
# ============================================================

class TLRLAgent:

    """
    Two-Lane Reinforcement Learning controller.

    Maintains:
        - active decision lane
        - inactive counterfactual memory
        - Q-values
        - visit statistics

    PERFORMANCE OPTIMIZATION:
        The original nearest-state Python scan has been
        replaced by an exact cKDTree lookup.

    The RL logic itself is unchanged.
    """

    def __init__(self):

        self.q = {}
        self.active_lane = {}
        self.inactive_memory = {}
        self.visit_counts = {}

        # ----------------------------------------------------
        # PERFORMANCE INDEX
        # ----------------------------------------------------

        self._state_order = []
        self._state_index = {}
        self._state_tree = None
        self._tree_points = None

    def _rebuild_state_tree(self):

        if not self._state_order:
            self._state_tree = None
            self._tree_points = None
            return

        self._tree_points = np.asarray(
            self._state_order,
            dtype=np.float64,
        )

        self._state_tree = cKDTree(
            self._tree_points
        )

    def _ensure_state(self, state_key):

        if state_key not in self.q:

            self.q[state_key] = np.zeros(
                len(ACTIONS),
                dtype=np.float64,
            )

            self.visit_counts[state_key] = 0

            # Keep insertion order exactly as
            # the original dictionary did.
            self._state_index[
                state_key
            ] = len(self._state_order)

            self._state_order.append(
                state_key
            )

            # Exact nearest-neighbour index.
            self._rebuild_state_tree()

    def q_values(self, state_key):

        if state_key in self.q:
            return self.q[state_key]

        if len(self.q) > 0:

            target = np.asarray(
                state_key,
                dtype=np.float64,
            )

            # ------------------------------------------------
            # EXACT nearest-state lookup
            # ------------------------------------------------
            #
            # This replaces:
            #
            # min(
            #     self.q.keys(),
            #     key=lambda x:
            #         np.sum(
            #             (np.array(x) - target)**2
            #         )
            # )
            #
            # cKDTree performs the nearest search in
            # compiled code instead of repeatedly executing
            # Python distance calculations.
            # ------------------------------------------------

            distance, index = (
                self._state_tree.query(
                    target,
                    k=1,
                )
            )

            # Preserve the original tie behaviour as closely
            # as possible: find all states at the same
            # distance and select the earliest dictionary
            # insertion.
            candidate_indices = (
                self._state_tree.query_ball_point(
                    target,
                    r=float(distance) + 1e-12,
                )
            )

            if len(candidate_indices) == 1:

                nearest = self._state_order[
                    int(index)
                ]

            else:

                nearest_index = min(
                    candidate_indices,
                    key=lambda i: i,
                )

                nearest = self._state_order[
                    nearest_index
                ]

            return self.q[
                nearest
            ].copy()

        self._ensure_state(
            state_key
        )

        return self.q[state_key]

    def choose(
        self,
        state_key,
        allowed,
        epsilon=0.0,
    ):

        self._ensure_state(
            state_key
        )

        q = self.q_values(
            state_key
        )

        if (
            epsilon > 0.0
            and random.random() < epsilon
        ):

            selected = random.choice(
                allowed
            )

        else:

            selected = max(
                allowed,
                key=lambda action:
                    q[action],
            )

        self.active_lane[
            state_key
        ] = selected

        self.inactive_memory[
            state_key
        ] = {
            action: float(q[action])
            for action in allowed
            if action != selected
        }

        return selected

    def update(
        self,
        state_key,
        action,
        reward,
        next_state_key,
        allowed_next,
    ):

        self._ensure_state(
            state_key
        )

        next_q = self.q_values(
            next_state_key
        )

        best_next = max(
            next_q[action]
            for action in allowed_next
        )

        td_target = (
            reward
            + GAMMA * best_next
        )

        self.q[state_key][action] += (
            ALPHA
            * (
                td_target
                - self.q[state_key][action]
            )
        )

        self.visit_counts[
            state_key
        ] += 1


# ============================================================
# 7. RD3P AGENT
# ============================================================

class RD3PAgent:

    """
    Reward Deficit Driven Projection controller.

    Uses projected cumulative reward,
    confidence bounds and dynamic target
    recalibration.
    """

    def __init__(
        self,
        target_return=30.0,
        tolerance_buffer=5.0,
    ):

        self.q = {}
        self.counts = {}

        self.target_0 = (
            target_return
            - tolerance_buffer
        )

        self.current_target = (
            self.target_0
        )

        self.smoothed_q = {}

        # ----------------------------------------------------
        # PERFORMANCE INDEX
        # ----------------------------------------------------

        self._state_order = []
        self._state_index = {}
        self._state_tree = None
        self._tree_points = None

    def _rebuild_state_tree(self):

        if not self._state_order:
            self._state_tree = None
            self._tree_points = None
            return

        self._tree_points = np.asarray(
            self._state_order,
            dtype=np.float64,
        )

        self._state_tree = cKDTree(
            self._tree_points
        )

    def _ensure_state(self, state_key):

        if state_key not in self.q:

            self.q[state_key] = np.zeros(
                len(ACTIONS),
                dtype=np.float64,
            )

        if state_key not in self.counts:

            self.counts[state_key] = np.zeros(
                len(ACTIONS),
                dtype=np.int32,
            )

        if state_key not in self.smoothed_q:

            self.smoothed_q[state_key] = np.zeros(
                len(ACTIONS),
                dtype=np.float64,
            )

        # Only add to the nearest-state index once.
        if state_key not in self._state_index:

            self._state_index[
                state_key
            ] = len(self._state_order)

            self._state_order.append(
                state_key
            )

            self._rebuild_state_tree()

    def q_values(self, state_key):

        if state_key in self.q:
            return self.q[state_key]

        if len(self.q) > 0:

            target = np.asarray(
                state_key,
                dtype=np.float64,
            )

            distance, index = (
                self._state_tree.query(
                    target,
                    k=1,
                )
            )

            candidate_indices = (
                self._state_tree.query_ball_point(
                    target,
                    r=float(distance) + 1e-12,
                )
            )

            if len(candidate_indices) == 1:

                nearest = self._state_order[
                    int(index)
                ]

            else:

                nearest_index = min(
                    candidate_indices,
                    key=lambda i: i,
                )

                nearest = self._state_order[
                    nearest_index
                ]

            return self.q[
                nearest
            ].copy()

        self._ensure_state(
            state_key
        )

        return self.q[state_key]

    def choose(
        self,
        state_key,
        allowed,
        steps_left=1,
        accum_reward=0.0,
    ):

        self._ensure_state(
            state_key
        )

        q = self.q_values(
            state_key
        )

        counts = self.counts[
            state_key
        ]

        best_arm = max(
            allowed,
            key=lambda action:
                q[action],
        )

        smoothed_q = self.smoothed_q[
            state_key
        ]

        projected_return = (
            accum_reward
            + (
                smoothed_q[best_arm]
                * max(1, steps_left)
            )
        )

        if (
            projected_return
            >= self.current_target
        ):

            return best_arm

        total_visits = (
            np.sum(counts) + 1
        )

        confidence_bounds = (
            q
            + 2.0
            * np.sqrt(
                np.log(
                    total_visits
                    + 1e-5
                )
                / (
                    counts
                    + 1e-5
                )
            )
        )

        projected_exploration = (
            accum_reward
            + (
                confidence_bounds
                * max(1, steps_left)
            )
        )

        candidates = [
            action
            for action in allowed
            if projected_exploration[action]
            >= self.current_target
        ]

        if candidates:

            return max(
                candidates,
                key=lambda action:
                    confidence_bounds[action],
            )

        best_possible = np.max(
            [
                projected_exploration[action]
                for action in allowed
            ]
        )

        recal_factor = min(
            1.0,
            max(
                0.01,
                best_possible
                / max(
                    self.target_0,
                    1e-6,
                ),
            ),
        )

        self.current_target = (
            recal_factor
            * self.target_0
        )

        return max(
            allowed,
            key=lambda action:
                confidence_bounds[action],
        )

    def update(
        self,
        state_key,
        action,
        reward,
        next_state_key,
        allowed_next,
    ):

        self._ensure_state(
            state_key
        )

        next_q = self.q_values(
            next_state_key
        )

        best_next = max(
            next_q[action]
            for action in allowed_next
        )

        td_target = (
            reward
            + GAMMA * best_next
        )

        self.q[state_key][action] += (
            ALPHA
            * (
                td_target
                - self.q[state_key][action]
            )
        )

        self.counts[
            state_key
        ][action] += 1

        self.smoothed_q[
            state_key
        ][action] = (
            0.8
            * self.smoothed_q[
                state_key
            ][action]
            + 0.2 * reward
        )


# ============================================================
# 8. RL TRAINING
# ============================================================

tl_agent = TLRLAgent()
rd3p_agent = RD3PAgent()

TRAINING_EPISODES = int(
    os.environ.get(
        "RL_TRAINING_EPISODES",
        "3500",
    )
)

print(
    "Executing Dynamic Adaptive Dual RL Policy Training..."
)

rng = np.random.default_rng(42)


for episode in range(
    TRAINING_EPISODES
):

    inventory = float(
        rng.uniform(
            2000,
            24000,
        )
    )

    demand = float(
        rng.uniform(
            8000,
            22000,
        )
    )

    weather = float(
        rng.uniform(
            0.0,
            1.0,
        )
    )

    route = float(
        rng.uniform(
            0.1,
            1.0,
        )
    )

    capacity = float(
        rng.uniform(
            0.1,
            1.0,
        )
    )

    trend = float(
        rng.uniform(
            -0.20,
            0.50,
        )
    )

    priority = float(
        rng.uniform(
            0.10,
            1.0,
        )
    )

    accumulated_rd3p_reward = 0.0

    steps = 6

    for step in range(steps):

        state = make_state(
            inventory,
            demand,
            weather,
            route,
            capacity,
            trend,
            priority,
        )

        state_key = discretize(
            state
        )

        allowed = feasible_actions(
            inventory,
            demand,
            route,
            capacity,
        )

        epsilon = max(
            0.02,
            0.30
            * (
                1.0
                - episode
                / max(
                    TRAINING_EPISODES,
                    1,
                )
            ),
        )

        action_tl = tl_agent.choose(
            state_key,
            allowed,
            epsilon=epsilon,
        )

        action_rd3p = rd3p_agent.choose(
            state_key,
            allowed,
            steps_left=(
                steps - step
            ),
            accum_reward=(
                accumulated_rd3p_reward
            ),
        )

        coverage = (
            inventory
            / max(
                demand,
                1.0,
            )
        )

        shortage = max(
            0.0,
            1.0 - coverage,
        )

        reward = (
            10.0
            - 45.0 * shortage
            - 15.0
            * max(
                0.0,
                0.70 - route,
            )
            - 12.0
            * max(
                0.0,
                0.70 - capacity,
            )
        )

        # Dynamic behavioural incentives.

        if (
            coverage < 0.60
            and action_tl
            in [
                ACTION_IDS[
                    "FORWARD REPLENISHMENT"
                ],
                ACTION_IDS[
                    "PRIORITY REPLENISHMENT"
                ],
            ]
        ):

            reward += (
                28.0
                + 10.0 * priority
            )

        if (
            route < 0.55
            and action_tl
            == ACTION_IDS[
                "ALTERNATE ROUTE"
            ]
        ):

            reward += 25.0

        if (
            capacity < 0.55
            and action_tl
            == ACTION_IDS[
                "RESERVE CAPACITY"
            ]
        ):

            reward += 22.0

        accumulated_rd3p_reward += reward

        consumption_factor = (
            float(
                rng.uniform(
                    0.10,
                    0.22,
                )
            )
            * (
                1.0
                + 0.15 * weather
            )
        )

        inventory = max(
            0.0,
            inventory
            - consumption_factor * demand,
        )

        demand = max(
            1000.0,
            demand
            * (
                1.0
                + 0.03 * trend
            ),
        )

        next_state = make_state(
            inventory,
            demand,
            weather,
            route,
            capacity,
            trend,
            priority,
        )

        next_state_key = discretize(
            next_state
        )

        next_allowed = feasible_actions(
            inventory,
            demand,
            route,
            capacity,
        )

        tl_agent.update(
            state_key,
            action_tl,
            reward,
            next_state_key,
            next_allowed,
        )

        rd3p_agent.update(
            state_key,
            action_rd3p,
            reward,
            next_state_key,
            next_allowed,
        )


print(
    f"RL training complete: "
    f"{TRAINING_EPISODES:,} episodes"
)


# ============================================================
# 9. GIS MAP ENGINE
# ============================================================

def build_tactical_gis_map(
    theater_key,
    route_health,
    chosen_action,
    sector_stock_ratio,
):

    theater = THEATERS.get(
        theater_key,
        THEATERS[
            "LADAKH_NORTHERN_AXIS"
        ],
    )

    map_object = folium.Map(
        location=theater["center"],
        zoom_start=theater["zoom"],
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # --------------------------------------------------------
    # Primary route
    # --------------------------------------------------------

    primary_color = (
        "#6f8f45"
        if route_health >= 0.55
        else "#b23b3b"
    )

    folium.PolyLine(
        theater["primary_route"],
        color=primary_color,
        weight=5,
        opacity=0.90,
        tooltip=(
            "PRIMARY SUPPLY CORRIDOR // "
            f"INTEGRITY {route_health * 100:.1f}%"
        ),
    ).add_to(map_object)

    # --------------------------------------------------------
    # Alternate route
    # --------------------------------------------------------

    if (
        route_health < 0.55
        or chosen_action == "ALTERNATE ROUTE"
    ):

        folium.PolyLine(
            theater["alternate_route"],
            color="#c69a3a",
            weight=5,
            dash_array="8, 8",
            opacity=0.95,
            tooltip=(
                "ALTERNATE CORRIDOR // "
                "REROUTE AVAILABLE"
            ),
        ).add_to(map_object)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for code, info in theater[
        "nodes"
    ].items():

        node_color = "#607d4d"

        if (
            code == "POST_SIERRA"
            and sector_stock_ratio < 0.55
        ):

            node_color = "#b43c3c"

        elif (
            code == "NODE_TRANSIT"
            and route_health < 0.55
        ):

            node_color = "#c69a3a"

        folium.CircleMarker(
            location=[
                info["lat"],
                info["lon"],
            ],
            radius=8,
            color=node_color,
            fill=True,
            fill_color=node_color,
            fill_opacity=0.95,
            weight=2,
            popup=(
                f"<b>{info['name']}</b>"
                f"<br>ROLE: {info['role']}"
                f"<br>NODE: {code}"
            ),
        ).add_to(map_object)

    # --------------------------------------------------------
    # Fit map
    # --------------------------------------------------------

    all_points = (
        theater["primary_route"]
        + theater["alternate_route"]
    )

    map_object.fit_bounds(
        [
            [
                min(
                    point[0]
                    for point in all_points
                ),
                min(
                    point[1]
                    for point in all_points
                ),
            ],
            [
                max(
                    point[0]
                    for point in all_points
                ),
                max(
                    point[1]
                    for point in all_points
                ),
            ],
        ]
    )

    return map_object._repr_html_()


# ============================================================
# 10. DECISION ENGINE
# ============================================================

def execute_logistics_matrix(
    theater_key,
    month_idx,
    inventory,
    operational_surge,
    weather,
    route,
    capacity,
    trend,
    priority,
):

    # --------------------------------------------------------
    # Demand prediction
    # --------------------------------------------------------

    demand = forecast_demand(
        int(month_idx),
        float(operational_surge),
        float(trend),
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state_vector = make_state(
        inventory,
        demand,
        weather,
        route,
        capacity,
        trend,
        priority,
    )

    state_key = discretize(
        state_vector
    )

    # --------------------------------------------------------
    # Feasibility
    # --------------------------------------------------------

    allowed = feasible_actions(
        inventory,
        demand,
        route,
        capacity,
    )

    # --------------------------------------------------------
    # RL policies
    # --------------------------------------------------------

    tl_q = tl_agent.q_values(
        state_key
    )

    rd_q = rd3p_agent.q_values(
        state_key
    )

    # --------------------------------------------------------
    # Score normalization
    # --------------------------------------------------------

    def normalize_scores(q_values):

        subset = np.array(
            [
                q_values[action]
                for action in allowed
            ],
            dtype=np.float64,
        )

        spread = np.ptp(
            subset
        )

        if spread < 1e-9:

            return {
                action:
                    1.0 / len(allowed)
                for action in allowed
            }

        minimum = np.min(
            subset
        )

        return {
            action:
                float(
                    (
                        q_values[action]
                        - minimum
                    )
                    / spread
                )
            for action in allowed
        }

    tl_normalized = normalize_scores(
        tl_q
    )

    rd_normalized = normalize_scores(
        rd_q
    )

    # --------------------------------------------------------
    # 50 / 50 ensemble
    # --------------------------------------------------------

    ensemble = {
        action:
            0.50
            * tl_normalized[action]
            + 0.50
            * rd_normalized[action]

        for action in allowed
    }

    final_action = ACTIONS[
        max(
            allowed,
            key=lambda action:
                ensemble[action],
        )
    ]

    tactical_choice = ACTIONS[
        max(
            allowed,
            key=lambda action:
                tl_normalized[action],
        )
    ]

    strategic_choice = ACTIONS[
        max(
            allowed,
            key=lambda action:
                rd_normalized[action],
        )
    ]

    # --------------------------------------------------------
    # Telemetry
    # --------------------------------------------------------

    coverage_ratio = (
        float(inventory)
        / max(
            float(demand),
            1.0,
        )
    )

    shortage = max(
        0.0,
        float(demand)
        - float(inventory),
    )

    risk_score = (
        0.40
        * min(
            1.0,
            max(
                0.0,
                1.0 - coverage_ratio,
            )
            / 0.70,
        )
        + 0.20 * float(weather)
        + 0.20
        * (
            1.0
            - float(route)
        )
        + 0.15
        * (
            1.0
            - float(capacity)
        )
        + 0.05
        * float(priority)
    )

    if risk_score >= 0.65:
        risk_tier = "CRITICAL"
    elif risk_score >= 0.45:
        risk_tier = "HIGH"
    elif risk_score >= 0.25:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # --------------------------------------------------------
    # Scoring table
    # --------------------------------------------------------

    score_rows = []

    for action_id, action_name in enumerate(
        ACTIONS
    ):

        if action_id in allowed:

            score_rows.append(
                [
                    action_name,
                    round(
                        tl_normalized[
                            action_id
                        ],
                        4,
                    ),
                    round(
                        rd_normalized[
                            action_id
                        ],
                        4,
                    ),
                    round(
                        ensemble[
                            action_id
                        ],
                        4,
                    ),
                    "FEASIBLE",
                ]
            )

        else:

            score_rows.append(
                [
                    action_name,
                    None,
                    None,
                    None,
                    "FILTERED",
                ]
            )

    score_df = pd.DataFrame(
        score_rows,
        columns=[
            "ACTION",
            "TL-RL",
            "RD3P",
            "ENSEMBLE",
            "STATUS",
        ],
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    map_html = build_tactical_gis_map(
        theater_key,
        float(route),
        final_action,
        coverage_ratio,
    )

    # --------------------------------------------------------
    # Output strings
    # --------------------------------------------------------

    directive_text = (
        f"## ◼ DISPATCH DIRECTIVE\n\n"
        f"# {final_action}"
    )

    telemetry_summary = (
        f"### LOGISTICS TELEMETRY\n\n"
        f"**FORECAST DEMAND**  \n"
        f"`{demand:,.0f} UNITS`\n\n"
        f"**ON-HAND STOCK**  \n"
        f"`{float(inventory):,.0f} UNITS`\n\n"
        f"**COVERAGE RATIO**  \n"
        f"`{coverage_ratio:.2f}`\n\n"
        f"**PROJECTED SHORTAGE**  \n"
        f"`{shortage:,.0f} UNITS`"
    )

    risk_summary = (
        f"### THREAT / RISK ASSESSMENT\n\n"
        f"**STATUS**  \n"
        f"`{risk_tier}`\n\n"
        f"**RISK INDEX**  \n"
        f"`{risk_score:.3f}`\n\n"
        f"**ROUTE INTEGRITY**  \n"
        f"`{float(route) * 100:.1f}%`\n\n"
        f"**CAPACITY HEADROOM**  \n"
        f"`{float(capacity) * 100:.1f}%`\n\n"
        f"**SECTOR PRIORITY**  \n"
        f"`{float(priority) * 100:.1f}%`"
    )

    consensus_text = (
        f"### POLICY CONSENSUS\n\n"
        f"**TL-RL POLICY**  \n"
        f"`{tactical_choice}`\n\n"
        f"**RD3P POLICY**  \n"
        f"`{strategic_choice}`\n\n"
        f"**FINAL CONSENSUS**  \n"
        f"`{'UNANIMOUS' if tactical_choice == strategic_choice else '50/50 ENSEMBLE'}`"
    )

    system_status = (
        "ONLINE"
        if len(allowed) > 0
        else "DEGRADED"
    )

    status_text = (
        f"**SYSTEM STATUS:** `{system_status}`   "
        f"**THEATER:** `{theater_key}`   "
        f"**FEASIBLE ACTIONS:** `{len(allowed)}/{len(ACTIONS)}`"
    )

    return (
        directive_text,
        telemetry_summary,
        risk_summary,
        consensus_text,
        score_df,
        map_html,
        status_text,
    )


# ============================================================
# 11. MILITARY COMMAND CONSOLE FRONTEND
# ============================================================

tactical_css = r"""

/* =========================================================
   GLOBAL
   ========================================================= */

:root {

    --bg: #080b08;
    --bg-2: #0c100c;

    --panel: #111711;
    --panel-2: #151b15;

    --border: #293329;

    --text: #c7d0c2;
    --muted: #7e8979;

    --green: #8fa66d;
    --green-bright: #a9c47f;

    --amber: #c59b4c;
    --red: #bd4d4d;

    --grid: rgba(139, 160, 112, 0.055);
}


/* =========================================================
   PAGE
   ========================================================= */

body,
.gradio-container {

    background:
        linear-gradient(
            rgba(8, 11, 8, 0.97),
            rgba(8, 11, 8, 0.97)
        ),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 31px,
            var(--grid) 32px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 31px,
            var(--grid) 32px
        ) !important;

    color: var(--text) !important;

    font-family:
        "IBM Plex Mono",
        "JetBrains Mono",
        "Lucida Console",
        Consolas,
        monospace !important;

    max-width: 1600px !important;
    margin: auto !important;
}


/* =========================================================
   REMOVE GENERIC ROUNDED LOOK
   ========================================================= */

.gradio-container * {

    border-radius: 2px !important;
}


/* =========================================================
   TOP HEADER
   ========================================================= */

.command-header {

    background:
        linear-gradient(
            90deg,
            #101610,
            #182018,
            #101610
        ) !important;

    border:
        1px solid #394737 !important;

    border-left:
        4px solid var(--green) !important;

    padding:
        18px 22px !important;

    margin-bottom:
        10px !important;

    box-shadow:
        0 0 24px
        rgba(0, 0, 0, 0.45) !important;
}


.command-title {

    color:
        #d5ddcf !important;

    font-size:
        25px !important;

    font-weight:
        800 !important;

    letter-spacing:
        3px !important;

    margin-bottom:
        5px !important;
}


.command-subtitle {

    color:
        var(--muted) !important;

    font-size:
        11px !important;

    letter-spacing:
        2px !important;
}


/* =========================================================
   STATUS BAR
   ========================================================= */

.status-bar {

    background:
        #0d120d !important;

    border:
        1px solid var(--border) !important;

    padding:
        8px 14px !important;

    color:
        var(--green-bright) !important;

    font-size:
        11px !important;

    letter-spacing:
        1.4px !important;

    margin-bottom:
        10px !important;
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-header {

    color:
        #aeb9a5 !important;

    font-size:
        11px !important;

    font-weight:
        800 !important;

    letter-spacing:
        2px !important;

    text-transform:
        uppercase !important;

    border-bottom:
        1px solid #303b30 !important;

    padding-bottom:
        7px !important;

    margin-bottom:
        9px !important;
}


/* =========================================================
   PANELS
   ========================================================= */

.command-panel {

    background:
        linear-gradient(
            145deg,
            #111711,
            #0d120d
        ) !important;

    border:
        1px solid var(--border) !important;

    padding:
        12px !important;

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,0.015),
        0 8px 24px
        rgba(0,0,0,0.22) !important;
}


/* =========================================================
   INPUTS
   ========================================================= */

input,
textarea,
button,
select {

    font-family:
        "IBM Plex Mono",
        "JetBrains Mono",
        "Lucida Console",
        monospace !important;
}


input,
textarea,
.gr-input,
.gr-dropdown,
.gr-number,
.gr-slider {

    background:
        #090d09 !important;

    color:
        #cbd4c5 !important;

    border:
        1px solid #303a30 !important;
}


/* =========================================================
   LABELS
   ========================================================= */

label {

    color:
        #889486 !important;

    font-size:
        10px !important;

    letter-spacing:
        1px !important;

    text-transform:
        uppercase !important;
}


/* =========================================================
   EXECUTE BUTTON
   ========================================================= */

.execute-button {

    background:
        linear-gradient(
            180deg,
            #34442e,
            #263322
        ) !important;

    color:
        #d7e1d0 !important;

    border:
        1px solid #65765a !important;

    font-size:
        13px !important;

    font-weight:
        900 !important;

    letter-spacing:
        2px !important;

    min-height:
        48px !important;

    text-transform:
        uppercase !important;

    box-shadow:
        0 0 16px
        rgba(119, 148, 92, 0.08) !important;
}


.execute-button:hover {

    background:
        linear-gradient(
            180deg,
            #40543a,
            #2e3d29
        ) !important;

    border-color:
        #839876 !important;
}


/* =========================================================
   DIRECTIVE PANEL
   ========================================================= */

.directive-panel {

    background:
        linear-gradient(
            135deg,
            #111811,
            #0a0e0a
        ) !important;

    border:
        1px solid #3d4c38 !important;

    border-left:
        5px solid var(--green) !important;

    min-height:
        130px !important;

    padding:
        18px !important;

    box-shadow:
        inset 0 0 30px
        rgba(100, 140, 70, 0.025) !important;
}


.directive-panel h2 {

    color:
        #7f9368 !important;

    font-size:
        12px !important;

    letter-spacing:
        2px !important;
}


.directive-panel h1 {

    color:
        #d5dfcf !important;

    font-size:
        25px !important;

    letter-spacing:
        2px !important;

    text-transform:
        uppercase !important;
}


/* =========================================================
   TELEMETRY CARDS
   ========================================================= */

.telemetry-card {

    background:
        #0c110c !important;

    border:
        1px solid #293329 !important;

    padding:
        14px !important;

    min-height:
        190px !important;
}


.telemetry-card h3 {

    color:
        #7e8d74 !important;

    font-size:
        10px !important;

    letter-spacing:
        1.7px !important;
}


/* =========================================================
   RISK PANEL
   ========================================================= */

.risk-card {

    background:
        linear-gradient(
            145deg,
            #15120d,
            #0e0f0b
        ) !important;

    border:
        1px solid #493e29 !important;

    padding:
        14px !important;
}


/* =========================================================
   CONSENSUS PANEL
   ========================================================= */

.consensus-card {

    background:
        linear-gradient(
            145deg,
            #0e140e,
            #0a0d0a
        ) !important;

    border:
        1px solid #34432f !important;

    padding:
        14px !important;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

.dataframe {

    background:
        #090d09 !important;

    border:
        1px solid #303a30 !important;
}


table {

    font-family:
        "IBM Plex Mono",
        "JetBrains Mono",
        monospace !important;

    font-size:
        11px !important;
}


th {

    background:
        #1b231a !important;

    color:
        #a9b89e !important;

    text-transform:
        uppercase !important;

    letter-spacing:
        1px !important;
}


td {

    background:
        #0c100c !important;

    color:
        #aeb9a8 !important;
}


/* =========================================================
   MAP
   ========================================================= */

.map-panel {

    border:
        1px solid #303a30 !important;

    background:
        #080b08 !important;

    padding:
        5px !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

.command-footer {

    border-top:
        1px solid #293329 !important;

    margin-top:
        12px !important;

    padding-top:
        8px !important;

    color:
        #586255 !important;

    font-size:
        9px !important;

    letter-spacing:
        1.2px !important;

    text-align:
        center !important;
}


/* =========================================================
   MARKDOWN
   ========================================================= */

.markdown-text {

    color:
        #adb7a9 !important;
}


code {

    color:
        #b8cc9e !important;

    background:
        #0a0e0a !important;

    border:
        1px solid #283227 !important;

    padding:
        2px 5px !important;
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {

    width:
        8px;
}


::-webkit-scrollbar-track {

    background:
        #080b08;
}


::-webkit-scrollbar-thumb {

    background:
        #303a2e;

    border-radius:
        0 !important;
}


::-webkit-scrollbar-thumb:hover {

    background:
        #46533f;
}

"""


# ============================================================
# 12. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    css=tactical_css,
    title="Tactical Predictive Logistics Command",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
    ),
) as app:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="command-header">
            <div class="command-title">
                ◈ TACTICAL PREDICTIVE LOGISTICS MANAGEMENT SYSTEM
            </div>

            <div class="command-subtitle">
                AUTONOMOUS LOGISTICS DECISION SUPPORT //
                TL-RL + RD3P CONSENSUS //
                PREDICTIVE DEMAND INTELLIGENCE
            </div>
        </div>
        """
    )

    status_display = gr.Markdown(
        """
        **SYSTEM STATUS:** `STANDBY` &nbsp;&nbsp;
        **DECISION ENGINE:** `READY` &nbsp;&nbsp;
        **RL POLICY:** `LOADED` &nbsp;&nbsp;
        **GIS:** `ONLINE`
        """,
        elem_classes=[
            "status-bar"
        ],
    )

    # --------------------------------------------------------
    # MAIN COMMAND AREA
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # LEFT: INPUT CONSOLE
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes=[
                "command-panel"
            ],
        ):

            gr.Markdown(
                "### ▣ OPERATIONAL TELEMETRY INPUT",
                elem_classes=[
                    "section-header"
                ],
            )

            in_theater = gr.Dropdown(
                choices=list(
                    THEATERS.keys()
                ),
                value="LADAKH_NORTHERN_AXIS",
                label="THEATER / SECTOR",
            )

            in_month = gr.Slider(
                minimum=0,
                maximum=5,
                value=0,
                step=1,
                label="FORECAST HORIZON / MONTH",
            )

            in_inv = gr.Number(
                value=11000,
                label="CURRENT STOCK / UNITS",
            )

            in_surge = gr.Slider(
                minimum=0.8,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="OPERATIONAL TEMPO / SURGE",
            )

            in_weather = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.30,
                step=0.05,
                label="WEATHER / TERRAIN FRICTION",
            )

            in_route = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.90,
                step=0.05,
                label="PRIMARY ROUTE INTEGRITY",
            )

            in_cap = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.85,
                step=0.05,
                label="FLEET CAPACITY HEADROOM",
            )

            in_trend = gr.Slider(
                minimum=-0.20,
                maximum=0.50,
                value=0.05,
                step=0.05,
                label="DEMAND ESCALATION TREND",
            )

            in_pri = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.75,
                step=0.05,
                label="SECTOR STRATEGIC PRIORITY",
            )

            btn_execute = gr.Button(
                "▶ EXECUTE DECISION MATRIX",
                variant="primary",
                elem_classes=[
                    "execute-button"
                ],
            )

        # ====================================================
        # RIGHT: COMMAND OUTPUT
        # ====================================================

        with gr.Column(
            scale=2,
        ):

            out_directive = gr.Markdown(
                """
                ## ◼ DISPATCH DIRECTIVE

                # AWAITING TELEMETRY
                """,
                elem_classes=[
                    "directive-panel"
                ],
            )

            with gr.Row():

                out_telemetry = gr.Markdown(
                    """
                    ### LOGISTICS TELEMETRY

                    Awaiting decision execution.
                    """,
                    elem_classes=[
                        "telemetry-card"
                    ],
                )

                out_risk = gr.Markdown(
                    """
                    ### THREAT / RISK ASSESSMENT

                    System awaiting telemetry.
                    """,
                    elem_classes=[
                        "risk-card"
                    ],
                )

                out_consensus = gr.Markdown(
                    """
                    ### POLICY CONSENSUS

                    TL-RL: STANDBY

                    RD3P: STANDBY
                    """,
                    elem_classes=[
                        "consensus-card"
                    ],
                )

            gr.Markdown(
                "### ▣ DECISION MATRIX / FEASIBILITY MASK",
                elem_classes=[
                    "section-header"
                ],
            )

            out_scores = gr.Dataframe(
                headers=[
                    "ACTION",
                    "TL-RL",
                    "RD3P",
                    "ENSEMBLE",
                    "STATUS",
                ],
                datatype=[
                    "str",
                    "number",
                    "number",
                    "number",
                    "str",
                ],
                value=pd.DataFrame(
                    columns=[
                        "ACTION",
                        "TL-RL",
                        "RD3P",
                        "ENSEMBLE",
                        "STATUS",
                    ]
                ),
                interactive=False,
                elem_classes=[
                    "command-panel"
                ],
            )

    # ========================================================
    # GIS
    # ========================================================

    gr.Markdown(
        "### ▣ SITUATIONAL CORRIDOR / NETWORK OVERVIEW",
        elem_classes=[
            "section-header"
        ],
    )

    with gr.Column(
        elem_classes=[
            "map-panel"
        ]
    ):

        out_map = gr.HTML(
            """
            <div style="
                height:520px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:#080b08;
                color:#596454;
                font-family:monospace;
                letter-spacing:2px;
                border:1px solid #293329;
            ">
                GIS SYSTEM STANDBY
            </div>
            """
        )

    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="command-footer">
            TACTICAL PREDICTIVE LOGISTICS SYSTEM //
            DECISION SUPPORT FRAMEWORK //
            RL POLICY EXECUTION + ML FORECASTING + GIS
        </div>
        """
    )

    # ========================================================
    # EVENT
    # ========================================================

    btn_execute.click(
        fn=execute_logistics_matrix,

        inputs=[
            in_theater,
            in_month,
            in_inv,
            in_surge,
            in_weather,
            in_route,
            in_cap,
            in_trend,
            in_pri,
        ],

        outputs=[
            out_directive,
            out_telemetry,
            out_risk,
            out_consensus,
            out_scores,
            out_map,
            status_display,
        ],
    )


# ============================================================
# 13. RENDER ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    # Render provides PORT automatically.
    port_value = os.environ.get(
        "PORT",
        "7860",
    )

    try:

        port = int(
            port_value
        )

    except ValueError:

        port = 7860

    print(
        "=" * 70
    )

    print(
        "TACTICAL PREDICTIVE LOGISTICS MANAGEMENT SYSTEM"
    )

    print(
        f"COMMAND CONSOLE STARTING ON 0.0.0.0:{port}"
    )

    print(
        f"RL TRAINING EPISODES: {TRAINING_EPISODES:,}"
    )

    print(
        "=" * 70
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
