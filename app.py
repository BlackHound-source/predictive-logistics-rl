# ============================================================
# PREDICTIVE LOGISTICS MANAGEMENT SYSTEM (TACTICAL EDITION)
# ============================================================
# DEPLOYMENT TARGET: Render / Hugging Face / Lightning AI / Local
# ============================================================

import os
import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 1. GRADIO IMPORT
# ============================================================

try:
    import gradio as gr
except ImportError:
    os.system("pip -q install -U gradio")
    import gradio as gr

# ============================================================
# 2. DIRECTORIES & PATHS (ADAPTIVE CLOUD / LOCAL PATHS)
# ============================================================

# Auto-detect if running on Kaggle or a standard cloud server / local
if Path("/kaggle/working").exists():
    BASE_DIR = Path("/kaggle/working")
else:
    BASE_DIR = Path(".")

FORECAST_DIR = BASE_DIR / "logistics_forecasting_results"
OUTPUT_DIR = BASE_DIR / "PREDICTIVE_LOGISTICS_RL_MVP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_FILE = FORECAST_DIR / "six_month_forecast.csv"
FORECAST_MODEL_FILE = FORECAST_DIR / "logistics_demand_forecaster.joblib"

# ============================================================
# 3. DEMAND FORECAST LOADER
# ============================================================

fallback_forecast = [
    15451.29,
    16742.14,
    16753.92,
    16945.08,
    17069.82,
    16969.06
]

def load_forecast():
    values = []
    if FORECAST_FILE.exists():
        try:
            data = pd.read_csv(FORECAST_FILE)
            preferred_columns = []
            for column in data.columns:
                name = str(column).lower()
                if any(term in name for term in ["forecast", "prediction", "demand"]):
                    preferred_columns.append(column)

            if len(preferred_columns) > 0:
                selected_column = preferred_columns[-1]
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_column = numeric_cols[-1] if len(numeric_cols) > 0 else None

            if selected_column is not None:
                values = pd.to_numeric(data[selected_column], errors="coerce").dropna().tolist()
        except Exception:
            pass

    if len(values) < 6:
        values = fallback_forecast

    return [float(max(1.0, x)) for x in values[:6]]

forecast_values = load_forecast()

# ============================================================
# 4. ACTION SPACE & GLOBAL CONFIGURATION
# ============================================================

ACTIONS = [
    "NORMAL SUPPLY",
    "FORWARD REPLENISHMENT",
    "PRIORITY REPLENISHMENT",
    "ALTERNATE ROUTE",
    "RESERVE CAPACITY"
]

ACTION_IDS = {action: index for index, action in enumerate(ACTIONS)}

STATE_BINS = 4
TRAIN_EPISODES = 12000
STEPS_PER_EPISODE = 6
ALPHA = 0.12
GAMMA = 0.93
TL_EPSILON_START = 0.30
TL_EPSILON_END = 0.02
RD3P_TOLERANCE = 0.50
RANDOM_SEED_TL = 123
RANDOM_SEED_RD3P = 456

# ============================================================
# 5. STATE REPRESENTATION & DECODER
# ============================================================

def clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))

def make_state(inventory, demand, weather, route, capacity, trend, priority):
    demand = max(float(demand), 1.0)
    inventory_ratio = clip01(float(inventory) / demand)
    demand_pressure = clip01(float(demand) / 25000.0)
    weather = clip01(weather)
    route = clip01(route)
    capacity = clip01(capacity)
    trend_normalized = clip01((float(trend) + 0.20) / 0.70)
    priority = clip01(priority)

    return np.array(
        [
            inventory_ratio,
            demand_pressure,
            weather,
            route,
            capacity,
            trend_normalized,
            priority
        ],
        dtype=np.float32
    )

def discretize(state):
    state = np.asarray(state, dtype=np.float32)
    bins = np.floor(state * STATE_BINS).astype(int)
    bins = np.clip(bins, 0, STATE_BINS - 1)
    return tuple(int(value) for value in bins)

def state_to_physical(state):
    state = np.asarray(state, dtype=np.float32)
    inventory_ratio = float(state[0])
    demand_pressure = float(state[1])
    weather = float(state[2])
    route = float(state[3])
    capacity = float(state[4])
    trend_normalized = float(state[5])
    priority = float(state[6])

    demand = max(1.0, demand_pressure * 25000.0)
    inventory = inventory_ratio * demand
    trend = trend_normalized * 0.70 - 0.20

    return {
        "inventory": inventory,
        "demand": demand,
        "weather": weather,
        "route": route,
        "capacity": capacity,
        "trend": trend,
        "priority": priority
    }

# ============================================================
# 6. FEASIBILITY MASK
# ============================================================

def feasible_actions(state):
    physical = state_to_physical(state)
    inventory = physical["inventory"]
    demand = physical["demand"]
    route = physical["route"]
    capacity = physical["capacity"]

    coverage = inventory / max(demand, 1.0)
    actions = list(range(len(ACTIONS)))

    if route >= 0.70:
        alternate_id = ACTION_IDS["ALTERNATE ROUTE"]
        if alternate_id in actions:
            actions.remove(alternate_id)

    if capacity >= 0.70:
        reserve_id = ACTION_IDS["RESERVE CAPACITY"]
        if reserve_id in actions:
            actions.remove(reserve_id)

    if coverage >= 1.10:
        for action_name in ["FORWARD REPLENISHMENT", "PRIORITY REPLENISHMENT"]:
            action_id = ACTION_IDS[action_name]
            if action_id in actions:
                actions.remove(action_id)
    elif coverage >= 0.95:
        priority_id = ACTION_IDS["PRIORITY REPLENISHMENT"]
        if priority_id in actions:
            actions.remove(priority_id)

    if len(actions) == 0:
        actions = [ACTION_IDS["NORMAL SUPPLY"]]

    return actions

# ============================================================
# 7. LOGISTICS DIGITAL TWIN ENVIRONMENT
# ============================================================

class LogisticsEnvironment:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        month = int(self.rng.integers(0, len(forecast_values)))
        demand = float(forecast_values[month])
        inventory_ratio = float(self.rng.uniform(0.15, 1.25))
        inventory = demand * inventory_ratio
        weather = float(self.rng.uniform(0.0, 1.0))
        route = float(self.rng.uniform(0.30, 1.0))
        capacity = float(self.rng.uniform(0.30, 1.0))
        trend = float(self.rng.uniform(-0.15, 0.35))
        priority = float(self.rng.uniform(0.20, 1.0))

        return {
            "inventory": inventory,
            "demand": demand,
            "weather": weather,
            "route": route,
            "capacity": capacity,
            "trend": trend,
            "priority": priority
        }

    def step(self, state, action):
        current = dict(state)
        demand_noise = float(self.rng.normal(0.0, 0.025))
        weather_factor = 1.0 + 0.12 * current["weather"]
        trend_factor = 1.0 + 0.35 * current["trend"]
        demand = current["demand"] * weather_factor * trend_factor * (1.0 + demand_noise)
        demand = max(1.0, demand)

        coverage_before = current["inventory"] / demand
        shortage_before = max(0.0, 1.0 - coverage_before)

        new_inventory = current["inventory"]
        new_route = current["route"]
        new_capacity = current["capacity"]

        if action == ACTION_IDS["FORWARD REPLENISHMENT"]:
            new_inventory += 0.18 * current["demand"] * current["capacity"]
            new_capacity -= 0.05
        elif action == ACTION_IDS["PRIORITY REPLENISHMENT"]:
            new_inventory += 0.30 * current["demand"] * current["capacity"]
            new_capacity -= 0.12
        elif action == ACTION_IDS["ALTERNATE ROUTE"]:
            new_route += 0.30
            new_capacity -= 0.05
        elif action == ACTION_IDS["RESERVE CAPACITY"]:
            new_capacity += 0.25

        new_inventory -= 0.12 * demand
        new_inventory = max(0.0, new_inventory)

        if current["weather"] > 0.65:
            new_route -= (current["weather"] - 0.65) * 0.20
        if current["weather"] > 0.75:
            new_capacity -= (current["weather"] - 0.75) * 0.15

        new_route = clip01(new_route)
        new_capacity = clip01(new_capacity)

        new_weather = clip01(current["weather"] + float(self.rng.normal(0.0, 0.04)))
        new_trend = float(np.clip(current["trend"] + float(self.rng.normal(0.0, 0.025)), -0.20, 0.50))
        new_demand = max(
            1.0,
            current["demand"] * np.clip(1.0 + 0.04 * current["trend"] + demand_noise * 0.30, 0.90, 1.10)
        )

        next_state = {
            "inventory": new_inventory,
            "demand": new_demand,
            "weather": new_weather,
            "route": new_route,
            "capacity": new_capacity,
            "trend": new_trend,
            "priority": current["priority"]
        }

        coverage_after = new_inventory / max(demand, 1.0)
        shortage_after = max(0.0, 1.0 - coverage_after)

        reward = 10.0
        reward -= 50.0 * shortage_after
        inventory_risk = max(0.0, 0.85 - coverage_after)
        reward -= 20.0 * inventory_risk
        route_risk = max(0.0, 0.70 - new_route)
        reward -= 20.0 * route_risk
        capacity_risk = max(0.0, 0.70 - new_capacity)
        reward -= 15.0 * capacity_risk

        if coverage_before < 0.55:
            if action == ACTION_IDS["PRIORITY REPLENISHMENT"]:
                reward += 30.0
            elif action == ACTION_IDS["FORWARD REPLENISHMENT"]:
                reward += 18.0
            elif action == ACTION_IDS["NORMAL SUPPLY"]:
                reward -= 20.0
        elif coverage_before < 0.80:
            if action == ACTION_IDS["FORWARD REPLENISHMENT"]:
                reward += 22.0
            elif action == ACTION_IDS["PRIORITY REPLENISHMENT"]:
                reward += 18.0
        else:
            if action == ACTION_IDS["NORMAL SUPPLY"]:
                reward += 10.0

        if current["route"] < 0.55 and action == ACTION_IDS["ALTERNATE ROUTE"]:
            reward += 25.0
        if current["capacity"] < 0.55 and action == ACTION_IDS["RESERVE CAPACITY"]:
            reward += 25.0
        if current["priority"] > 0.75 and coverage_before < 0.75:
            if action == ACTION_IDS["PRIORITY REPLENISHMENT"]:
                reward += 10.0

        info = {
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "shortage_before": shortage_before,
            "shortage_after": shortage_after
        }

        return next_state, float(reward), info

# ============================================================
# 8. RL MODELS (WITH NEAREST-NEIGHBOR RESOLUTION)
# ============================================================

class TLRLAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        self.alpha = alpha
        self.gamma = gamma
        self.q = {}
        self.active_lane = {}
        self.inactive_memory = {}
        self.visit_counts = {}

    def q_values(self, state_key):
        if state_key in self.q:
            return self.q[state_key]

        if len(self.q) > 0:
            target = np.array(state_key, dtype=np.float32)
            nearest_key = min(
                self.q.keys(),
                key=lambda k: np.sum((np.array(k, dtype=np.float32) - target) ** 2)
            )
            return self.q[nearest_key].copy()

        self.q[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)
        return self.q[state_key]

    def choose(self, state, epsilon=0.0):
        state_key = discretize(state)
        q = self.q_values(state_key)
        actions = feasible_actions(state)

        if epsilon > 0.0 and random.random() < epsilon:
            selected = random.choice(actions)
        else:
            selected = max(actions, key=lambda action: q[action])

        self.active_lane[state_key] = selected
        self.inactive_memory[state_key] = {
            action: float(q[action])
            for action in actions
            if action != selected
        }
        return selected

    def update(self, state, action, reward, next_state):
        state_key = discretize(state)
        next_key = discretize(next_state)

        if state_key not in self.q:
            self.q[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)
        if next_key not in self.q:
            self.q[next_key] = np.zeros(len(ACTIONS), dtype=np.float64)

        q = self.q[state_key]
        next_q = self.q[next_key]
        next_actions = feasible_actions(next_state)
        best_next = max(next_q[action_id] for action_id in next_actions)
        target = reward + self.gamma * best_next

        q[action] += self.alpha * (target - q[action])
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1


class RD3PAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA, tolerance=RD3P_TOLERANCE):
        self.alpha = alpha
        self.gamma = gamma
        self.tolerance = tolerance
        self.q = {}
        self.target = 12.0
        self.target_history = []
        self.visit_counts = {}

    def q_values(self, state_key):
        if state_key in self.q:
            return self.q[state_key]

        if len(self.q) > 0:
            target = np.array(state_key, dtype=np.float32)
            nearest_key = min(
                self.q.keys(),
                key=lambda k: np.sum((np.array(k, dtype=np.float32) - target) ** 2)
            )
            return self.q[nearest_key].copy()

        self.q[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)
        return self.q[state_key]

    def choose(self, state, steps_left=1):
        state_key = discretize(state)
        q = self.q_values(state_key)
        actions = feasible_actions(state)
        values = np.array([q[action_id] for action_id in actions], dtype=np.float64)

        best_value = float(np.max(values))
        projected_return = best_value * max(1, steps_left)
        deficit = self.target - projected_return

        if deficit > self.tolerance:
            uncertainty = 0.10 + min(1.50, deficit / (abs(self.target) + 1.0))
            noise = np.random.normal(0.0, uncertainty, len(actions))
            scores = values + noise
        else:
            scores = values

        selected_index = int(np.argmax(scores))
        return actions[selected_index]

    def update(self, state, action, reward, next_state):
        state_key = discretize(state)
        next_key = discretize(next_state)

        if state_key not in self.q:
            self.q[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)
        if next_key not in self.q:
            self.q[next_key] = np.zeros(len(ACTIONS), dtype=np.float64)

        q = self.q[state_key]
        next_q = self.q[next_key]
        next_actions = feasible_actions(next_state)
        best_next = max(next_q[action_id] for action_id in next_actions)
        target = reward + self.gamma * best_next

        q[action] += self.alpha * (target - q[action])
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1

# ============================================================
# 9. TRAINING INITIALIZATION & PERSISTENCE
# ============================================================

print("Executing Adaptive RL Policy Training...")

random.seed(RANDOM_SEED_TL)
np.random.seed(RANDOM_SEED_TL)
tl_agent = TLRLAgent()
tl_environment = LogisticsEnvironment(seed=RANDOM_SEED_TL)

for episode in range(TRAIN_EPISODES):
    state = tl_environment.reset()
    epsilon_decay = episode / max(1, TRAIN_EPISODES - 1)
    epsilon = TL_EPSILON_START * (1.0 - epsilon_decay) + TL_EPSILON_END * epsilon_decay

    for step in range(STEPS_PER_EPISODE):
        state_vector = make_state(
            state["inventory"], state["demand"], state["weather"],
            state["route"], state["capacity"], state["trend"], state["priority"]
        )
        action = tl_agent.choose(state_vector, epsilon=epsilon)
        next_state, reward, _ = tl_environment.step(state, action)
        next_vector = make_state(
            next_state["inventory"], next_state["demand"], next_state["weather"],
            next_state["route"], next_state["capacity"], next_state["trend"], next_state["priority"]
        )
        tl_agent.update(state_vector, action, reward, next_vector)
        state = next_state

random.seed(RANDOM_SEED_RD3P)
np.random.seed(RANDOM_SEED_RD3P)
rd3p_agent = RD3PAgent()
rd3p_environment = LogisticsEnvironment(seed=RANDOM_SEED_RD3P)

for episode in range(TRAIN_EPISODES):
    state = rd3p_environment.reset()

    for step in range(STEPS_PER_EPISODE):
        state_vector = make_state(
            state["inventory"], state["demand"], state["weather"],
            state["route"], state["capacity"], state["trend"], state["priority"]
        )
        action = rd3p_agent.choose(state_vector, steps_left=(STEPS_PER_EPISODE - step))
        next_state, reward, _ = rd3p_environment.step(state, action)
        next_vector = make_state(
            next_state["inventory"], next_state["demand"], next_state["weather"],
            next_state["route"], next_state["capacity"], next_state["trend"], next_state["priority"]
        )
        rd3p_agent.update(state_vector, action, reward, next_vector)
        state = next_state

# ============================================================
# 10. DECISION ENGINE & RISK MODEL
# ============================================================

def normalized_scores(q_values, actions):
    values = np.array([float(q_values[action]) for action in actions], dtype=np.float64)
    if len(values) == 1:
        return {actions[0]: 1.0}
    minimum, maximum = float(np.min(values)), float(np.max(values))
    spread = maximum - minimum
    if spread < 1e-12:
        normalized = np.ones(len(values)) / len(values)
    else:
        normalized = (values - minimum) / spread
    return {action: float(score) for action, score in zip(actions, normalized)}

def calculate_risk(inventory, demand, weather, route, capacity, priority):
    coverage = inventory / max(demand, 1.0)
    shortage_ratio = max(0.0, 1.0 - coverage)
    inventory_component = min(1.0, shortage_ratio / 0.70)
    score = (
        0.40 * inventory_component
        + 0.20 * weather
        + 0.20 * (1.0 - route)
        + 0.15 * (1.0 - capacity)
        + 0.05 * priority
    )
    if score >= 0.70:
        level = "CRITICAL"
    elif score >= 0.45:
        level = "HIGH"
    elif score >= 0.25:
        level = "MEDIUM"
    else:
        level = "LOW"
    return float(score), level, float(coverage), float(shortage_ratio)

def run_logistics_decision(month, inventory, demand, weather, route, capacity, trend, priority):
    month = int(np.clip(month, 0, len(forecast_values) - 1))
    inventory = max(0.0, float(inventory))
    demand = max(1.0, float(demand))
    weather, route, capacity, priority = clip01(weather), clip01(route), clip01(capacity), clip01(priority)
    trend = float(np.clip(trend, -0.20, 0.50))

    state_vector = make_state(inventory, demand, weather, route, capacity, trend, priority)
    state_key = discretize(state_vector)
    allowed = feasible_actions(state_vector)

    tl_q = tl_agent.q_values(state_key)
    rd_q = rd3p_agent.q_values(state_key)

    tl_scores = normalized_scores(tl_q, allowed)
    rd_scores = normalized_scores(rd_q, allowed)

    combined_scores = {action: 0.50 * tl_scores[action] + 0.50 * rd_scores[action] for action in allowed}

    tl_action_id = max(allowed, key=lambda a: tl_scores[a])
    rd_action_id = max(allowed, key=lambda a: rd_scores[a])
    final_action_id = max(allowed, key=lambda a: combined_scores[a])

    coverage = inventory / max(demand, 1.0)
    shortage = max(0.0, demand - inventory)
    shortage_percent = (shortage / max(demand, 1.0)) * 100.0
    risk_score, risk_level, coverage, shortage_ratio = calculate_risk(
        inventory, demand, weather, route, capacity, priority
    )

    policy_agreement = (tl_action_id == rd_action_id)
    sorted_scores = sorted(combined_scores.values(), reverse=True)
    score_gap = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 1.0

    if policy_agreement and score_gap >= 0.20:
        confidence = "HIGH"
    elif score_gap >= 0.10:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    reasons = []
    if inventory < demand:
        reasons.append("Inventory deficit detected.")
    else:
        reasons.append("Sufficient inventory buffer.")
    if weather >= 0.70:
        reasons.append("Severe meteorological resistance.")
    if route < 0.55:
        reasons.append("Primary line-of-communication degraded.")
    if capacity < 0.55:
        reasons.append("Transport throughput constrained.")
    if trend > 0.15:
        reasons.append("Demand escalation pattern active.")
    if priority >= 0.75:
        reasons.append("Strategic sector priority designated.")

    policy_rows = []
    for action_id, action_name in enumerate(ACTIONS):
        if action_id in allowed:
            policy_rows.append([
                action_name,
                round(tl_scores[action_id], 4),
                round(rd_scores[action_id], 4),
                round(combined_scores[action_id], 4),
                "FEASIBLE"
            ])
        else:
            policy_rows.append([action_name, None, None, None, "FILTERED"])

    policy_score_df = pd.DataFrame(
        policy_rows,
        columns=["Action", "Tactical Policy", "Strategic Policy", "Ensemble Score", "Operational Status"]
    )

    return {
        "month": month,
        "base_forecast": forecast_values[month],
        "demand": demand,
        "inventory": inventory,
        "weather": weather,
        "route": route,
        "capacity": capacity,
        "trend": trend,
        "priority": priority,
        "coverage": coverage,
        "shortage": shortage,
        "shortage_percent": shortage_percent,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "allowed": allowed,
        "tl_action": ACTIONS[tl_action_id],
        "rd_action": ACTIONS[rd_action_id],
        "final_action": ACTIONS[final_action_id],
        "policy_agreement": policy_agreement,
        "score_gap": score_gap,
        "confidence": confidence,
        "reason": " ".join(reasons),
        "policy_score_df": policy_score_df
    }

# ============================================================
# 11. GRADIO FRONTEND (MUTED TACTICAL PALETTE // NO TL / RD3P)
# ============================================================

tactical_css = """
:root {
    --tac-bg: #111411;
    --tac-surface: #181d18;
    --tac-border: #2a332a;
    --tac-border-focus: #424e42;
    --tac-text-main: #bcc6b9;
    --tac-text-dim: #737f70;
    --tac-accent: #8b9986;
    --tac-code-bg: #0d100d;
}

body, .gradio-container {
    background-color: var(--tac-bg) !important;
    color: var(--tac-text-main) !important;
    font-family: "Lucida Console", Monaco, "Courier New", monospace !important;
    max-width: 1350px !important;
}

div[class*="block"], div[class*="panel"], div[class*="form"] {
    background-color: var(--tac-surface) !important;
    border: 1px solid var(--tac-border) !important;
    border-radius: 2px !important;
}

h1, h2, h3, h4 {
    color: var(--tac-text-main) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 700 !important;
}

.decision_terminal {
    background-color: var(--tac-code-bg) !important;
    border-left: 3px solid var(--tac-accent) !important;
    padding: 12px !important;
    margin-bottom: 12px !important;
}

.decision_terminal h1 {
    font-size: 22px !important;
    margin: 0 !important;
    color: var(--tac-text-main) !important;
}

button.primary-btn {
    background-color: #242c24 !important;
    color: #cad6c7 !important;
    border: 1px solid #3b463a !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    border-radius: 0px !important;
    letter-spacing: 2px !important;
}

button.primary-btn:hover {
    background-color: #2d382d !important;
    border-color: #556353 !important;
}

input, textarea {
    background-color: #141714 !important;
    color: #cad6c7 !important;
    border: 1px solid var(--tac-border) !important;
    border-radius: 0px !important;
    font-family: inherit !important;
}

input:focus {
    border-color: var(--tac-border-focus) !important;
}

table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 12px !important;
}

th {
    background-color: #141914 !important;
    color: var(--tac-text-dim) !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--tac-border) !important;
}

td {
    border: 1px solid #1f251f !important;
    color: var(--tac-text-main) !important;
}
"""

def ui_calculate_decision(month, inv, dem, weather, route, cap, trend, priority):
    res = run_logistics_decision(month, inv, dem, weather, route, cap, trend, priority)

    decision_header = f"# [DISPATCH DIRECTIVE] : {res['final_action']}"

    threat_telemetry = (
        f"**RISK CLASSIFICATION** : {res['risk_level']}\n\n"
        f"**THREAT INDEX**        : {res['risk_score']:.4f}\n\n"
        f"**DIRECTIVE CONFIDENCE** : {res['confidence']}"
    )

    supply_telemetry = (
        f"**FORECAST DEMAND**     : {res['base_forecast']:,.0f} UNITS\n\n"
        f"**OPERATIONAL DEMAND**  : {res['demand']:,.0f} UNITS\n\n"
        f"**STOCK ON HAND**       : {res['inventory']:,.0f} UNITS\n\n"
        f"**COVERAGE RATIO**      : {res['coverage']:.2f} PERIODS\n\n"
        f"**SHORTAGE MARGIN**     : {res['shortage']:,.0f} UNITS ({res['shortage_percent']:.1f}%)\n\n"
        f"**ROUTE HEALTH**        : {res['route'] * 100:.1f}%\n\n"
        f"**CAPACITY HEADROOM**   : {res['capacity'] * 100:.1f}%"
    )

    policy_telemetry = (
        f"### MODEL CONSENSUS EVALUATION\n\n"
        f"**TACTICAL DIRECTIVE**   : {res['tl_action']}\n\n"
        f"**STRATEGIC DIRECTIVE**  : {res['rd_action']}\n\n"
        f"**UNANIMOUS AGREEMENT**  : {'YES' if res['policy_agreement'] else 'NO (RESOLVED VIA ENSEMBLE)'}\n\n"
        f"**MARGIN SEPARATION**    : {res['score_gap']:.4f}\n\n"
        f"**OPERATIONAL RATIONALE**: {res['reason']}"
    )

    state_matrix = pd.DataFrame([
        ["MONTH INDEX", res["month"] + 1],
        ["DEMAND (BASE)", f"{res['base_forecast']:,.2f}"],
        ["DEMAND (EFFECTIVE)", f"{res['demand']:,.2f}"],
        ["INVENTORY ON HAND", f"{res['inventory']:,.2f}"],
        ["COVERAGE RATIO", f"{res['coverage']:.3f}"],
        ["WEATHER STRESS INDEX", f"{res['weather']:.3f}"],
        ["PRIMARY ROUTE INTEGRITY", f"{res['route']:.3f}"],
        ["TRANSPORT CAPACITY", f"{res['capacity']:.3f}"],
        ["DEMAND TREND COEFFICIENT", f"{res['trend']:+.3f}"],
        ["SECTOR PRIORITY TIER", f"{res['priority']:.3f}"]
    ], columns=["OPERATIONAL PARAMETER", "METRIC"])

    return decision_header, threat_telemetry, supply_telemetry, policy_telemetry, state_matrix, res["policy_score_df"]

with gr.Blocks(css=tactical_css, title="Logistics Tactical Command Console") as app:
    gr.Markdown("## LOGISTICS TACTICAL COMMAND // DIRECTIVE CONTROLLER")
    gr.Markdown("SYSTEM ARCHITECTURE: DEMAND PROJECTION + DUAL-POLICY ENSEMBLE DECISION ENGINE")

    with gr.Tab("COMMAND DISPATCH"):
        with gr.Row():
            with gr.Column():
                ui_month = gr.Slider(0, 5, value=0, step=1, label="FORECAST MONTH [0-5]")
                ui_inv = gr.Number(value=15000, label="ON-HAND INVENTORY")
                ui_dem = gr.Number(value=float(forecast_values[0]), label="DEMAND VALUE")
                ui_trend = gr.Slider(-0.20, 0.50, value=0.00, step=0.01, label="DEMAND TREND")
            with gr.Column():
                ui_weather = gr.Slider(0.0, 1.0, value=0.20, step=0.01, label="WEATHER STRESS")
                ui_route = gr.Slider(0.0, 1.0, value=1.00, step=0.01, label="ROUTE INTEGRITY")
                ui_cap = gr.Slider(0.0, 1.0, value=1.00, step=0.01, label="FLEET CAPACITY")
                ui_pri = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="SECTOR PRIORITY")

        run_btn = gr.Button("EXECUTE DECISION MATRIX", elem_classes=["primary-btn"])

        out_decision = gr.Markdown(elem_classes=["decision_terminal"])

        with gr.Row():
            out_threat = gr.Markdown()
            out_supply = gr.Markdown()

        out_policy = gr.Markdown()

        with gr.Row():
            out_matrix = gr.Dataframe(interactive=False, label="INPUT TELEMETRY STATE")
            out_scores = gr.Dataframe(interactive=False, label="POLICY SCORING MATRIX")

        ui_month.change(
            fn=lambda m: float(forecast_values[int(np.clip(m, 0, len(forecast_values)-1))]),
            inputs=ui_month,
            outputs=ui_dem
        )

        run_btn.click(
            fn=ui_calculate_decision,
            inputs=[ui_month, ui_inv, ui_dem, ui_weather, ui_route, ui_cap, ui_trend, ui_pri],
            outputs=[out_decision, out_threat, out_supply, out_policy, out_matrix, out_scores]
        )

    with gr.Tab("SYSTEM AUDIT RECORDS"):
        forecast_df = pd.DataFrame({"MONTH": np.arange(1, len(forecast_values)+1), "DEMAND": forecast_values})
        gr.Dataframe(forecast_df, label="ACTIVE SIX-MONTH BASELINE PROJECTION")

# ============================================================
# 12. CLOUD & LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False
    )
