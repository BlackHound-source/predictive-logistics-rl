"""Reinforcement Learning (RL) Controller."""

import random
from config import ACTIONS, ALPHA, GAMMA
from agents.base import KDTreeIndexedAgent


class TLRLAgent(KDTreeIndexedAgent):
    def __init__(self):
        super().__init__()
        self.active_lane = {}
        self.inactive_memory = {}
        self.visit_counts = {}

    def _ensure_state(self, state_key):
        if state_key not in self.q:
            self._ensure_base_state(state_key)
            self.visit_counts[state_key] = 0

    def q_values(self, state_key):
        return self.get_nearest_q(state_key)

    def choose(self, state_key, allowed, epsilon=0.0):
        self._ensure_state(state_key)
        q = self.q_values(state_key)

        if epsilon > 0.0 and random.random() < epsilon:
            selected = random.choice(allowed)
        else:
            selected = max(allowed, key=lambda a: q[a])

        self.active_lane[state_key] = selected
        self.inactive_memory[state_key] = {a: float(q[a]) for a in allowed if a != selected}
        return selected

    def update(self, state_key, action, reward, next_state_key, allowed_next):
        self._ensure_state(state_key)
        next_q = self.q_values(next_state_key)
        best_next = max(next_q[a] for a in allowed_next)
        td_target = reward + GAMMA * best_next

        self.q[state_key][action] += ALPHA * (td_target - self.q[state_key][action])
        self.visit_counts[state_key] += 1
