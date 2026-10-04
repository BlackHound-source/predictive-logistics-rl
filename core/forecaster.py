"""Multivariate polynomial demand forecaster."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class DemandForecaster:
    def __init__(self):
        self.model = self._train_initial_model()

    def _train_initial_model(self):
        np.random.seed(42)
        t = np.arange(1, 37).reshape(-1, 1)
        base = (
            13500.0
            + 120.0 * t.flatten()
            + 1400.0 * np.sin(t.flatten() / 2.5)
            + np.random.normal(0, 250, 36)
        )
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
        model.fit(t, base)
        return model

    def predict(self, month_idx: int, operational_surge: float = 1.0, trend: float = 0.0) -> float:
        t_eval = np.array([[37 + month_idx]])
        raw_pred = self.model.predict(t_eval)[0]
        trend_adjusted = raw_pred * (1.0 + float(trend) * 0.45)
        return float(max(1000.0, trend_adjusted * operational_surge))


forecaster = DemandForecaster()
