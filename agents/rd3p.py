"""Reward Driven Controller."""

import numpy as np
from config import ACTIONS, ALPHA, GAMMA
from agents.base import KDTreeIndexedAgent


class RD3PAgent(KDTreeIndexedAgent):
    def __init__(self, target_return=30.0, tolerance_buffer=5.0):
        super().__init__()
        self.counts = {}
        self.smoothed_q = {}
        self.target_0 = target_return - tolerance_buffer
        self.current_target = self.target_0

    def _ensure_state(self, state_key):
        if state_key not in self.q:
            self._ensure_base_state(state_key)
            self.counts[state_key] = np.zeros(len(ACTIONS), dtype=np.int32)
            self.smoothed_q[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)

    def q_values(self, state_key):
        return self.get_nearest_q(state_key)

    def choose(self, state_key, allowed, steps_left=1, accum_reward=0.0):
        self._ensure_state(state_key)
        q = self.q_values(state_key)
        counts = self.counts[state_key]
        best_arm = max(allowed, key=lambda a: q[a])

        projected_return = accum_reward + (self.smoothed_q[state_key][best_arm] * max(1, steps_left))
        if projected_return >= self.current_target:
            return best_arm

        total_visits = np.sum(counts) + 1
        confidence_bounds = q + 2.0 * np.sqrt(np.log(total_visits + 1e-5) / (counts + 1e-5))
        projected_exp = accum_reward + (confidence_bounds * max(1, steps_left))

        candidates = [a for a in allowed if projected_exp[a] >= self.current_target]
        if candidates:
            return max(candidates, key=lambda a: confidence_bounds[a])

        best_possible = np.max([projected_exp[a] for a in allowed])
        recal_factor = min(1.0, max(0.01, best_possible / max(self.target_0, 1e-6)))
        self.current_target = recal_factor * self.target_0

        return max(allowed, key=lambda a: confidence_bounds[a])

    def update(self, state_key, action, reward, next_state_key, allowed_next):
        self._ensure_state(state_key)
        next_q = self.q_values(next_state_key)
        best_next = max(next_q[a] for a in allowed_next)
        td_target = reward + GAMMA * best_next

        self.q[state_key][action] += ALPHA * (td_target - self.q[state_key][action])
        self.counts[state_key][action] += 1
        self.smoothed_q[state_key][action] = (
            0.8 * self.smoothed_q[state_key][action] + 0.2 * reward
        )
