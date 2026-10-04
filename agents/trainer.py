"""Dual-policy offline warm-up and reinforcement learning training loop."""

import os
import numpy as np
from config import ACTIONS, ACTION_IDS
from core.environment import make_state, discretize, feasible_actions
from agents.tl_rl import TLRLAgent
from agents.rd3p import RD3PAgent


def train_agents(episodes: int = None):
    if episodes is None:
        episodes = int(os.environ.get("RL_TRAINING_EPISODES", "3500"))

    print(f"Executing Dynamic Dual RL Training: {episodes:,} episodes...")
    tl_agent = TLRLAgent()
    rd3p_agent = RD3PAgent()
    rng = np.random.default_rng(42)

    for ep in range(episodes):
        inv = float(rng.uniform(2000, 24000))
        dem = float(rng.uniform(8000, 22000))
        w = float(rng.uniform(0.0, 1.0))
        rt = float(rng.uniform(0.1, 1.0))
        cap = float(rng.uniform(0.1, 1.0))
        trnd = float(rng.uniform(-0.20, 0.50))
        pri = float(rng.uniform(0.10, 1.0))

        accum_reward = 0.0
        steps = 6

        for s in range(steps):
            state = make_state(inv, dem, w, rt, cap, trnd, pri)
            state_k = discretize(state)
            allowed = feasible_actions(inv, dem, rt, cap)

            epsilon = max(0.02, 0.30 * (1.0 - ep / max(episodes, 1)))
            a_tl = tl_agent.choose(state_k, allowed, epsilon=epsilon)
            a_rd = rd3p_agent.choose(state_k, allowed, steps_left=(steps - s), accum_reward=accum_reward)

            cov = inv / max(dem, 1.0)
            shortage = max(0.0, 1.0 - cov)
            reward = (
                10.0
                - 45.0 * shortage
                - 15.0 * max(0.0, 0.70 - rt)
                - 12.0 * max(0.0, 0.70 - cap)
            )

            if cov < 0.60 and a_tl in [ACTION_IDS["FORWARD REPLENISHMENT"], ACTION_IDS["PRIORITY REPLENISHMENT"]]:
                reward += 28.0 + 10.0 * pri
            if rt < 0.55 and a_tl == ACTION_IDS["ALTERNATE ROUTE"]:
                reward += 25.0
            if cap < 0.55 and a_tl == ACTION_IDS["RESERVE CAPACITY"]:
                reward += 22.0

            accum_reward += reward
            inv = max(0.0, inv - float(rng.uniform(0.10, 0.22)) * (1.0 + 0.15 * w) * dem)
            dem = max(1000.0, dem * (1.0 + 0.03 * trnd))

            next_k = discretize(make_state(inv, dem, w, rt, cap, trnd, pri))
            next_allowed = feasible_actions(inv, dem, rt, cap)

            tl_agent.update(state_k, a_tl, reward, next_k, next_allowed)
            rd3p_agent.update(state_k, a_rd, reward, next_k, next_allowed)

    print(f"Policy convergence established across {episodes:,} steps.")
    return tl_agent, rd3p_agent
