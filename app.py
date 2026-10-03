# ============================================================
# TACTICAL PREDICTIVE LOGISTICS MANAGEMENT SYSTEM
# ============================================================
# Render-ready deployment
#
# Components:
#   1. Polynomial Temporal Demand Forecasting
#   2. Geospatial Digital Twin
#   3. TL-RL Agent
#   4. RD3P Agent
#   5. Feasibility Constraint Layer
#   6. 50/50 RL Ensemble
#   7. Gradio Command Console
#
# Deployment:
#   - Render compatible
#   - Uses Render's PORT automatically
#   - No Gradio share server
#   - No external map API key
#   - OpenStreetMap tiles
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


warnings.filterwarnings("ignore")


# ============================================================
# GRADIO
# ============================================================

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is not installed. "
        "Run: pip install -r requirements.txt"
    )


# ============================================================
# 1. NETWORK TOPOLOGY
# ============================================================

NODES = {

    "DEPOT_MAIN": {
        "name": "Rear Staging Base (RSB-01)",
        "lat": 34.1526,
        "lon": 77.5771,
        "role": "Supply Hub"
    },

    "NODE_TRANSIT": {
        "name": "Intermediate Logistics Node (ILN-North)",
        "lat": 34.3500,
        "lon": 77.6200,
        "role": "Transit Chokepoint"
    },

    "FOB_ALPHA": {
        "name": "Forward Operating Base (FOB-Alpha)",
        "lat": 34.6000,
        "lon": 77.7000,
        "role": "Forward Staging Depot"
    },

    "POST_SIERRA": {
        "name": "High-Altitude Forward Post (Sierra)",
        "lat": 34.8500,
        "lon": 77.7800,
        "role": "Terminal Sector"
    }
}


PRIMARY_ROUTE = [

    [34.1526, 77.5771],
    [34.3500, 77.6200],
    [34.6000, 77.7000],
    [34.8500, 77.7800]

]


ALTERNATE_ROUTE = [

    [34.1526, 77.5771],
    [34.2800, 77.4500],
    [34.5200, 77.5300],
    [34.8500, 77.7800]

]


# ============================================================
# 2. TEMPORAL DEMAND FORECASTING
# ============================================================

def train_temporal_demand_model():

    np.random.seed(42)

    t = np.arange(
        1,
        25
    ).reshape(
        -1,
        1
    )

    base_demand = (

        14000.0

        + 150.0 * t.flatten()

        + 1200.0
        * np.sin(
            t.flatten() / 2.0
        )

        + np.random.normal(
            0,
            300,
            24
        )
    )

    model = make_pipeline(

        PolynomialFeatures(
            degree=2
        ),

        Ridge(
            alpha=1.0
        )
    )

    model.fit(
        t,
        base_demand
    )

    return model


demand_model = (
    train_temporal_demand_model()
)


def forecast_demand(
    month_idx: int,
    operational_surge: float = 1.0
):

    t_eval = np.array(
        [[25 + month_idx]]
    )

    predicted = (
        demand_model
        .predict(t_eval)[0]
    )

    return float(
        max(
            1000.0,
            predicted
            * operational_surge
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

    "RESERVE CAPACITY"

]


ACTION_IDS = {

    action: index

    for index, action
    in enumerate(ACTIONS)

}


STATE_BINS = 4

ALPHA = 0.12

GAMMA = 0.93


# ============================================================
# UTILITIES
# ============================================================

def clip01(v):

    return float(
        np.clip(
            float(v),
            0.0,
            1.0
        )
    )


def make_state(
    inventory,
    demand,
    weather,
    route,
    capacity,
    trend,
    priority
):

    demand = max(
        float(demand),
        1.0
    )

    return np.array(

        [

            clip01(
                float(inventory)
                / demand
            ),

            clip01(
                float(demand)
                / 25000.0
            ),

            clip01(weather),

            clip01(route),

            clip01(capacity),

            clip01(
                (
                    float(trend)
                    + 0.20
                )
                / 0.70
            ),

            clip01(priority)

        ],

        dtype=np.float32

    )


def discretize(state):

    bins = np.floor(
        state * STATE_BINS
    ).astype(int)

    bins = np.clip(
        bins,
        0,
        STATE_BINS - 1
    )

    return tuple(
        int(v)
        for v in bins
    )


# ============================================================
# FEASIBILITY LAYER
# ============================================================

def feasible_actions(
    inventory,
    demand,
    route,
    capacity
):

    cov = (
        inventory
        / max(
            demand,
            1.0
        )
    )

    allowed = list(
        range(
            len(ACTIONS)
        )
    )

    # Primary corridor healthy
    if (

        route >= 0.70

        and ACTION_IDS[
            "ALTERNATE ROUTE"
        ] in allowed

    ):

        allowed.remove(
            ACTION_IDS[
                "ALTERNATE ROUTE"
            ]
        )


    # Fleet capacity healthy
    if (

        capacity >= 0.70

        and ACTION_IDS[
            "RESERVE CAPACITY"
        ] in allowed

    ):

        allowed.remove(
            ACTION_IDS[
                "RESERVE CAPACITY"
            ]
        )


    # Healthy stock
    if cov >= 1.10:

        for action in [

            "FORWARD REPLENISHMENT",
            "PRIORITY REPLENISHMENT"

        ]:

            aid = ACTION_IDS[action]

            if aid in allowed:

                allowed.remove(aid)


    # Moderate stock
    elif cov >= 0.95:

        aid = ACTION_IDS[
            "PRIORITY REPLENISHMENT"
        ]

        if aid in allowed:

            allowed.remove(aid)


    if not allowed:

        return [
            ACTION_IDS[
                "NORMAL SUPPLY"
            ]
        ]

    return allowed


# ============================================================
# 4. TL-RL AGENT
# ============================================================

class TLRLAgent:

    """
    Two-Lane Reinforcement Learning agent.

    Maintains an active policy lane while retaining
    inactive counterfactual action information.
    """

    def __init__(self):

        self.q = {}

        self.active_lane = {}

        self.inactive_memory = {}

        self.visit_counts = {}


    def _ensure_state(self, k):

        if k not in self.q:

            self.q[k] = np.zeros(
                len(ACTIONS),
                dtype=np.float64
            )

            self.visit_counts[k] = 0


    def q_values(self, k):

        if k in self.q:

            return self.q[k]


        if len(self.q) > 0:

            target = np.array(
                k,
                dtype=np.float32
            )

            nearest = min(

                self.q.keys(),

                key=lambda x:

                np.sum(

                    (
                        np.array(
                            x,
                            dtype=np.float32
                        )
                        - target
                    ) ** 2

                )
            )

            return self.q[
                nearest
            ].copy()


        self._ensure_state(k)

        return self.q[k]


    def choose(
        self,
        state_key,
        allowed,
        epsilon=0.0
    ):

        self._ensure_state(
            state_key
        )

        q = self.q_values(
            state_key
        )


        if (

            epsilon > 0.0

            and random.random()
            < epsilon

        ):

            selected = random.choice(
                allowed
            )

        else:

            selected = max(

                allowed,

                key=lambda a:
                q[a]

            )


        self.active_lane[
            state_key
        ] = selected


        self.inactive_memory[
            state_key
        ] = {

            a: float(q[a])

            for a in allowed

            if a != selected

        }


        return selected


    def update(
        self,
        k,
        a,
        r,
        next_k,
        allowed_next
    ):

        self._ensure_state(k)

        next_q = self.q_values(
            next_k
        )

        best_next = max(

            next_q[an]

            for an in allowed_next

        )


        self.q[k][a] += (

            ALPHA
            * (
                r
                + GAMMA * best_next
                - self.q[k][a]
            )

        )


        self.visit_counts[k] += 1


# ============================================================
# 5. RD3P AGENT
# ============================================================

class RD3PAgent:

    """
    Reward-Deficit-Driven Projection Policy.

    Uses projected returns, confidence bounds and
    dynamic target recalibration.
    """

    def __init__(
        self,
        target_return=30.0,
        tolerance_buffer=5.0
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


    def _ensure_state(self, k):

        if k not in self.q:

            self.q[k] = np.zeros(
                len(ACTIONS),
                dtype=np.float64
            )


        if k not in self.counts:

            self.counts[k] = np.zeros(
                len(ACTIONS),
                dtype=np.int32
            )


        if k not in self.smoothed_q:

            self.smoothed_q[k] = np.zeros(
                len(ACTIONS),
                dtype=np.float64
            )


    def q_values(self, k):

        if k in self.q:

            return self.q[k]


        if len(self.q) > 0:

            target = np.array(
                k,
                dtype=np.float32
            )

            nearest = min(

                self.q.keys(),

                key=lambda x:

                np.sum(

                    (
                        np.array(
                            x,
                            dtype=np.float32
                        )
                        - target
                    ) ** 2

                )
            )

            return self.q[
                nearest
            ].copy()


        self._ensure_state(k)

        return self.q[k]


    def choose(
        self,
        state_key,
        allowed,
        steps_left=1,
        accum_reward=0.0
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

            key=lambda a:
            q[a]

        )


        sm_q = self.smoothed_q[
            state_key
        ]


        projected_return = (

            accum_reward

            + (

                sm_q[best_arm]
                * max(
                    1,
                    steps_left
                )

            )

        )


        # Target achieved
        if (
            projected_return
            >= self.current_target
        ):

            return best_arm


        # Reward deficit
        total_visits = (
            np.sum(counts)
            + 1
        )


        confidence_bounds = (

            q

            + 2.0
            * np.sqrt(

                np.log(
                    total_visits
                    + 1e-5
                )

                /

                (
                    counts
                    + 1e-5
                )

            )

        )


        proj_explore = (

            accum_reward

            + (

                confidence_bounds
                * max(
                    1,
                    steps_left
                )

            )

        )


        candidates = [

            a

            for a in allowed

            if proj_explore[a]
            >= self.current_target

        ]


        if candidates:

            return max(

                candidates,

                key=lambda a:
                confidence_bounds[a]

            )


        # Dynamic recalibration
        best_possible = max(

            proj_explore[a]

            for a in allowed

        )


        recal_factor = min(

            1.0,

            max(

                0.01,

                best_possible
                / max(
                    self.target_0,
                    1e-6
                )

            )

        )


        self.current_target = (

            recal_factor
            * self.target_0

        )


        return max(

            allowed,

            key=lambda a:
            confidence_bounds[a]

        )


    def update(
        self,
        k,
        a,
        r,
        next_k,
        allowed_next
    ):

        self._ensure_state(k)

        next_q = self.q_values(
            next_k
        )


        best_next = max(

            next_q[an]

            for an in allowed_next

        )


        self.q[k][a] += (

            ALPHA

            * (

                r

                + GAMMA * best_next

                - self.q[k][a]

            )

        )


        self.counts[k][a] += 1


        self.smoothed_q[k][a] = (

            0.8
            * self.smoothed_q[k][a]

            + 0.2 * r

        )


# ============================================================
# INITIALIZE AGENTS
# ============================================================

tl_agent = TLRLAgent()

rd3p_agent = RD3PAgent()


# ============================================================
# 6. RL TRAINING
# ============================================================

print(
    "Executing Adaptive Dual RL Policy Training..."
)


rng = np.random.default_rng(
    42
)


# Render environment variable can control this.
# Default remains 3500 to preserve your original behavior.
TRAINING_EPISODES = int(

    os.environ.get(
        "RL_TRAINING_EPISODES",
        "3500"
    )

)


for ep in range(
    TRAINING_EPISODES
):

    inv = float(
        rng.uniform(
            3000,
            20000
        )
    )


    dem = float(
        rng.uniform(
            10000,
            18000
        )
    )


    weather = float(
        rng.uniform(
            0,
            1
        )
    )


    route = float(
        rng.uniform(
            0.2,
            1.0
        )
    )


    capacity = float(
        rng.uniform(
            0.3,
            1.0
        )
    )


    accum_r_rd = 0.0

    steps = 6


    for step in range(steps):

        state = make_state(

            inv,

            dem,

            weather,

            route,

            capacity,

            0.05,

            0.75

        )


        state_key = discretize(
            state
        )


        allowed = feasible_actions(

            inv,

            dem,

            route,

            capacity

        )


        epsilon = max(

            0.02,

            0.30
            * (

                1.0

                - ep
                / max(
                    TRAINING_EPISODES,
                    1
                )

            )

        )


        act_tl = tl_agent.choose(

            state_key,

            allowed,

            epsilon=epsilon

        )


        act_rd = rd3p_agent.choose(

            state_key,

            allowed,

            steps_left=(
                steps - step
            ),

            accum_reward=accum_r_rd

        )


        # ----------------------------------------------------
        # Environment feedback
        # ----------------------------------------------------

        coverage = (

            inv
            / max(
                dem,
                1.0
            )

        )


        shortage = max(

            0.0,

            1.0 - coverage

        )


        reward = (

            10.0

            - 45.0 * shortage

            - 15.0
            * max(
                0.0,
                0.70 - route
            )

            - 12.0
            * max(
                0.0,
                0.70 - capacity
            )

        )


        # Replenishment
        if coverage < 0.60:

            if act_tl in [

                ACTION_IDS[
                    "FORWARD REPLENISHMENT"
                ],

                ACTION_IDS[
                    "PRIORITY REPLENISHMENT"
                ]

            ]:

                reward += 28.0


        # Alternate route
        if (

            route < 0.55

            and act_tl
            == ACTION_IDS[
                "ALTERNATE ROUTE"
            ]

        ):

            reward += 25.0


        # Reserve capacity
        if (

            capacity < 0.55

            and act_tl
            == ACTION_IDS[
                "RESERVE CAPACITY"
            ]

        ):

            reward += 22.0


        accum_r_rd += reward


        # ----------------------------------------------------
        # State transition
        # ----------------------------------------------------

        inv = max(

            0.0,

            inv
            - 0.15 * dem

        )


        next_state = make_state(

            inv,

            dem,

            weather,

            route,

            capacity,

            0.05,

            0.75

        )


        next_key = discretize(
            next_state
        )


        next_allowed = feasible_actions(

            inv,

            dem,

            route,

            capacity

        )


        # ----------------------------------------------------
        # Update agents
        # ----------------------------------------------------

        tl_agent.update(

            state_key,

            act_tl,

            reward,

            next_key,

            next_allowed

        )


        rd3p_agent.update(

            state_key,

            act_rd,

            reward,

            next_key,

            next_allowed

        )


# ============================================================
# 7. GIS MAP
# ============================================================

def build_tactical_gis_map(

    route_health,

    chosen_action,

    sector_stock_ratio

):

    """
    Creates the GIS visualization.

    OpenStreetMap is used so the application does not require
    a CartoDB/Google Maps API key.
    """

    m = folium.Map(

        location=[
            34.4800,
            77.6500
        ],

        zoom_start=9,

        tiles="OpenStreetMap",

        control_scale=True

    )


    # --------------------------------------------------------
    # Primary route
    # --------------------------------------------------------

    primary_color = (

        "#3d7a42"

        if route_health >= 0.55

        else "#8f3e3e"

    )


    folium.PolyLine(

        PRIMARY_ROUTE,

        color=primary_color,

        weight=4,

        opacity=0.85,

        tooltip=(

            "Main Corridor // Health: "

            f"{route_health * 100:.1f}%"

        )

    ).add_to(m)


    # --------------------------------------------------------
    # Alternate route
    # --------------------------------------------------------

    if (

        route_health < 0.55

        or chosen_action
        == "ALTERNATE ROUTE"

    ):

        folium.PolyLine(

            ALTERNATE_ROUTE,

            color="#a38c44",

            weight=4,

            dash_array="6, 6",

            opacity=0.9,

            tooltip=(

                "TACTICAL BYPASS CORRIDOR "
                "(ACTIVE RE-ROUTE)"

            )

        ).add_to(m)


    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for code, info in NODES.items():

        node_color = "#4c6b4f"


        if (

            code == "POST_SIERRA"

            and sector_stock_ratio < 0.55

        ):

            node_color = "#994747"


        folium.CircleMarker(

            location=[

                info["lat"],

                info["lon"]

            ],

            radius=8,

            color=node_color,

            fill=True,

            fill_color=node_color,

            fill_opacity=0.9,

            popup=(

                f"<b>{info['name']}</b>"

                f"<br>Designation: "
                f"{info['role']}"

            )

        ).add_to(m)


    # --------------------------------------------------------
    # Fit map to routes
    # --------------------------------------------------------

    all_points = (

        PRIMARY_ROUTE
        + ALTERNATE_ROUTE

    )


    m.fit_bounds([

        [

            min(
                p[0]
                for p in all_points
            ),

            min(
                p[1]
                for p in all_points
            )

        ],

        [

            max(
                p[0]
                for p in all_points
            ),

            max(
                p[1]
                for p in all_points
            )

        ]

    ])


    return m._repr_html_()


# ============================================================
# 8. UNIFIED INFERENCE / DECISION ENGINE
# ============================================================

def execute_logistics_matrix(

    month_idx,

    inventory,

    operational_surge,

    weather,

    route,

    capacity,

    trend,

    priority

):

    # --------------------------------------------------------
    # ML demand forecast
    # --------------------------------------------------------

    demand = forecast_demand(

        int(month_idx),

        float(
            operational_surge
        )

    )


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state_vec = make_state(

        inventory,

        demand,

        weather,

        route,

        capacity,

        trend,

        priority

    )


    state_key = discretize(
        state_vec
    )


    # --------------------------------------------------------
    # Feasibility
    # --------------------------------------------------------

    allowed = feasible_actions(

        inventory,

        demand,

        route,

        capacity

    )


    # --------------------------------------------------------
    # Retrieve learned policies
    # --------------------------------------------------------

    tl_q = tl_agent.q_values(
        state_key
    )


    rd_q = rd3p_agent.q_values(
        state_key
    )


    # --------------------------------------------------------
    # Normalize policy scores
    # --------------------------------------------------------

    def norm_scores(q_arr):

        sub = np.array(

            [
                q_arr[a]
                for a in allowed
            ],

            dtype=np.float64

        )


        spread = np.ptp(
            sub
        )


        if spread < 1e-9:

            return {

                a:
                1.0 / len(allowed)

                for a
                in allowed

            }


        minimum = np.min(
            sub
        )


        return {

            a:

            float(

                (
                    q_arr[a]
                    - minimum
                )
                / spread

            )

            for a
            in allowed

        }


    tl_norm = norm_scores(
        tl_q
    )


    rd_norm = norm_scores(
        rd_q
    )


    # --------------------------------------------------------
    # 50/50 Ensemble
    # --------------------------------------------------------

    ensemble = {

        a:

        0.50 * tl_norm[a]

        + 0.50 * rd_norm[a]

        for a
        in allowed

    }


    # --------------------------------------------------------
    # Final action
    # --------------------------------------------------------

    final_action = ACTIONS[

        max(

            allowed,

            key=lambda a:
            ensemble[a]

        )

    ]


    tactical_choice = ACTIONS[

        max(

            allowed,

            key=lambda a:
            tl_norm[a]

        )

    ]


    strategic_choice = ACTIONS[

        max(

            allowed,

            key=lambda a:
            rd_norm[a]

        )

    ]


    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    coverage_ratio = (

        inventory
        / max(
            demand,
            1.0
        )

    )


    shortage = max(

        0.0,

        demand
        - inventory

    )


    risk_score = (

        0.40
        * min(

            1.0,

            max(
                0.0,
                1.0 - coverage_ratio
            )
            / 0.70

        )

        + 0.20 * weather

        + 0.20 * (
            1.0 - route
        )

        + 0.15 * (
            1.0 - capacity
        )

        + 0.05 * priority

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


    for aid, name in enumerate(
        ACTIONS
    ):

        if aid in allowed:

            score_rows.append([

                name,

                round(
                    tl_norm[aid],
                    4
                ),

                round(
                    rd_norm[aid],
                    4
                ),

                round(
                    ensemble[aid],
                    4
                ),

                "FEASIBLE"

            ])

        else:

            score_rows.append([

                name,

                None,

                None,

                None,

                "FILTERED"

            ])


    score_df = pd.DataFrame(

        score_rows,

        columns=[

            "Action",

            "Tactical Policy",

            "Strategic Policy",

            "Ensemble Score",

            "Operational Status"

        ]

    )


    # --------------------------------------------------------
    # GIS
    # --------------------------------------------------------

    map_html = build_tactical_gis_map(

        route,

        final_action,

        coverage_ratio

    )


    # --------------------------------------------------------
    # Text output
    # --------------------------------------------------------

    directive_text = (

        "# [DISPATCH DIRECTIVE] : "

        f"{final_action}"

    )


    telemetry_summary = (

        f"**FORECAST DEMAND (ML)** : "
        f"{demand:,.0f} UNITS\n\n"

        f"**ON-HAND STOCK**        : "
        f"{inventory:,.0f} UNITS\n\n"

        f"**COVERAGE RATIO**       : "
        f"{coverage_ratio:.2f} PERIODS\n\n"

        f"**PROJECTED SHORTAGE**   : "
        f"{shortage:,.0f} UNITS\n\n"

        f"**RISK ASSESSMENT**      : "
        f"{risk_tier} "
        f"(INDEX: {risk_score:.3f})"

    )


    consensus_text = (

        f"**TACTICAL DIRECTIVE**   : "
        f"{tactical_choice}\n\n"

        f"**STRATEGIC DIRECTIVE**  : "
        f"{strategic_choice}\n\n"

        f"**CONSENSUS STATE**       : "

        + (

            "UNANIMOUS"

            if tactical_choice
            == strategic_choice

            else
            "RESOLVED VIA 50/50 ENSEMBLE"

        )

    )


    return (

        directive_text,

        telemetry_summary,

        consensus_text,

        score_df,

        map_html

    )


# ============================================================
# 9. GRADIO UI
# ============================================================

tactical_css = """

:root {

    --tac-bg: #111411;

    --tac-surface: #181d18;

    --tac-border: #2a332a;

    --tac-text-main: #bcc6b9;

    --tac-accent: #8b9986;

}


body,
.gradio-container {

    background-color:
        var(--tac-bg)
        !important;

    color:
        var(--tac-text-main)
        !important;

    font-family:
        "Lucida Console",
        Monaco,
        monospace
        !important;

    max-width:
        1400px
        !important;

}


div[class*="block"] {

    background-color:
        var(--tac-surface)
        !important;

    border:
        1px solid
        var(--tac-border)
        !important;

    border-radius:
        2px
        !important;

}


.decision_terminal {

    background-color:
        #0d100d
        !important;

    border-left:
        3px solid
        var(--tac-accent)
        !important;

    padding:
        12px
        !important;

    margin-bottom:
        12px
        !important;

}


button.primary-btn {

    background-color:
        #242c24
        !important;

    color:
        #cad6c7
        !important;

    border:
        1px solid
        #3b463a
        !important;

    text-transform:
        uppercase
        !important;

    font-weight:
        700
        !important;

    letter-spacing:
        2px
        !important;

}

"""


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(

    css=tactical_css,

    title=
        "Tactical GIS Predictive Logistics"

) as app:

    gr.Markdown(

        "## PREDICTIVE LOGISTICS "
        "MANAGEMENT SYSTEM "
        "// COMMAND CONSOLE"

    )


    gr.Markdown(

        "TECHNOLOGY STACK: "
        "POLYNOMIAL TEMPORAL DEMAND "
        "FORECASTING + "
        "DUAL RL CONSENSUS + "
        "TACTICAL GIS ROUTE OPTIMIZATION"

    )


    with gr.Row():

        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        with gr.Column(
            scale=1
        ):

            gr.Markdown(
                "### OPERATIONAL TELEMETRY INPUTS"
            )


            in_month = gr.Slider(

                0,
                5,

                value=0,

                step=1,

                label=
                    "FORECAST HORIZON "
                    "[MONTH 0-5]"

            )


            in_inv = gr.Number(

                value=11000,

                label=
                    "CURRENT STOCK "
                    "(UNITS)"

            )


            in_surge = gr.Slider(

                0.8,
                2.0,

                value=1.0,

                step=0.1,

                label=
                    "OP TEMPO SURGE "
                    "MULTIPLIER"

            )


            in_weather = gr.Slider(

                0.0,
                1.0,

                value=0.30,

                step=0.05,

                label=
                    "TERRAIN / WEATHER "
                    "FRICTION"

            )


            in_route = gr.Slider(

                0.0,
                1.0,

                value=0.90,

                step=0.05,

                label=
                    "LINE-OF-COMMUNICATION "
                    "INTEGRITY"

            )


            in_cap = gr.Slider(

                0.0,
                1.0,

                value=0.85,

                step=0.05,

                label=
                    "FLEET CAPACITY "
                    "HEADROOM"

            )


            in_trend = gr.Slider(

                -0.20,
                0.50,

                value=0.05,

                step=0.05,

                label=
                    "DEMAND ESCALATION "
                    "TREND"

            )


            in_pri = gr.Slider(

                0.0,
                1.0,

                value=0.75,

                step=0.05,

                label=
                    "SECTOR STRATEGIC "
                    "PRIORITY"

            )


            btn_exec = gr.Button(

                "EXECUTE LOGISTICS DISPATCH",

                elem_classes=[
                    "primary-btn"
                ]

            )


        # ----------------------------------------------------
        # OUTPUTS
        # ----------------------------------------------------

        with gr.Column(
            scale=2
        ):

            out_directive = gr.Markdown(

                elem_classes=[
                    "decision_terminal"
                ]

            )


            with gr.Row():

                out_telemetry = (
                    gr.Markdown()
                )

                out_consensus = (
                    gr.Markdown()
                )


            out_scores = gr.Dataframe(

                label=
                    "ACTION SCORING MATRIX "
                    "(FEASIBILITY RESTRICTED)"

            )


            gr.Markdown(

                "### GIS SITUATIONAL "
                "CORRIDOR & CHOKEPOINT MAP"

            )


            out_map = gr.HTML(

                label=
                    "LIVE OPERATIONAL MAP"

            )


    # --------------------------------------------------------
    # EXECUTION EVENT
    # --------------------------------------------------------

    btn_exec.click(

        fn=execute_logistics_matrix,

        inputs=[

            in_month,

            in_inv,

            in_surge,

            in_weather,

            in_route,

            in_cap,

            in_trend,

            in_pri

        ],

        outputs=[

            out_directive,

            out_telemetry,

            out_consensus,

            out_scores,

            out_map

        ]

    )


# ============================================================
# 10. RENDER SERVER STARTUP
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "TACTICAL PREDICTIVE LOGISTICS "
        "MANAGEMENT SYSTEM"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Render provides PORT automatically.
    # --------------------------------------------------------

    port_value = os.environ.get(
        "PORT",
        "7860"
    )


    try:

        port = int(
            port_value
        )

    except ValueError:

        port = 7860


    print(
        f"Starting Gradio on port: {port}"
    )

    print(
        "Host: 0.0.0.0"
    )

    print(
        "Map: OpenStreetMap"
    )

    print(
        "External API key: NONE"
    )

    print(
        "Gradio share: DISABLED"
    )

    print("=" * 70)


    app.launch(

        server_name="0.0.0.0",

        server_port=port,

        share=False,

        show_error=True

    )
