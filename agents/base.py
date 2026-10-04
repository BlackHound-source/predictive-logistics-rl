"""Base class providing compiled cKDTree nearest-neighbour state lookups."""

import numpy as np
from scipy.spatial import cKDTree
from config import ACTIONS


class KDTreeIndexedAgent:
    def __init__(self):
        self.q = {}
        self._state_order = []
        self._state_index = {}
        self._state_tree = None

    def _rebuild_state_tree(self):
        if not self._state_order:
            self._state_tree = None
            return
        points = np.asarray(self._state_order, dtype=np.float64)
        self._state_tree = cKDTree(points)

    def _ensure_base_state(self, state_key):
        if state_key not in self.q:
            self.q[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)
            self._state_index[state_key] = len(self._state_order)
            self._state_order.append(state_key)
            self._rebuild_state_tree()

    def get_nearest_q(self, state_key):
        if state_key in self.q:
            return self.q[state_key]

        if len(self.q) > 0 and self._state_tree is not None:
            target = np.asarray(state_key, dtype=np.float64)
            distance, index = self._state_tree.query(target, k=1)
            candidate_indices = self._state_tree.query_ball_point(
                target, r=float(distance) + 1e-12
            )

            nearest_index = (
                int(index) if len(candidate_indices) == 1 else min(candidate_indices)
            )
            return self.q[self._state_order[nearest_index]].copy()

        self._ensure_base_state(state_key)
        return self.q[state_key]
