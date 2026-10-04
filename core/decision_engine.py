"""Consensus, normalization, and operational scoring pipeline."""

import numpy as np
import pandas as pd
from config import ACTIONS
from core.forecaster import forecaster
from core.environment import make_state, discretize, feasible_actions
from core.gis import build_tactical_gis_map


def _normalize(q_values, allowed):
    subset = np.array([q_values[a] for a in allowed], dtype=np.float64)
    spread = np.ptp(subset)
    if spread < 1e-9:
        return {a: 1.0 / len(allowed) for a in allowed}
    minimum = np.min(subset)
    return {a: float((q_values[a] - minimum) / spread) for a in allowed}


def evaluate_decision(
    tl_agent, rd3p_agent,
    theater_key, month_idx, inventory, operational_surge,
    weather, route, capacity, trend, priority
):
    demand = forecaster.predict(int(month_idx), float(operational_surge), float(trend))
    state_k = discretize(make_state(inventory, demand, weather, route, capacity, trend, priority))
    allowed = feasible_actions(inventory, demand, route, capacity)

    tl_norm = _normalize(tl_agent.q_values(state_k), allowed)
    rd_norm = _normalize(rd3p_agent.q_values(state_k), allowed)
    ensemble = {a: 0.50 * tl_norm[a] + 0.50 * rd_norm[a] for a in allowed}

    final_action = ACTIONS[max(allowed, key=lambda a: ensemble[a])]
    tactical_choice = ACTIONS[max(allowed, key=lambda a: tl_norm[a])]
    strategic_choice = ACTIONS[max(allowed, key=lambda a: rd_norm[a])]

    coverage_ratio = float(inventory) / max(float(demand), 1.0)
    shortage = max(0.0, float(demand) - float(inventory))

    risk_score = (
        0.40 * min(1.0, max(0.0, 1.0 - coverage_ratio) / 0.70)
        + 0.20 * float(weather)
        + 0.20 * (1.0 - float(route))
        + 0.15 * (1.0 - float(capacity))
        + 0.05 * float(priority)
    )

    if risk_score >= 0.65:
        risk_tier = "CRITICAL"
    elif risk_score >= 0.45:
        risk_tier = "HIGH"
    elif risk_score >= 0.25:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    score_rows = [
        [
            act_name,
            round(tl_norm[aid], 4) if aid in allowed else None,
            round(rd_norm[aid], 4) if aid in allowed else None,
            round(ensemble[aid], 4) if aid in allowed else None,
            "FEASIBLE" if aid in allowed else "FILTERED",
        ]
        for aid, act_name in enumerate(ACTIONS)
    ]

    score_df = pd.DataFrame(score_rows, columns=["ACTION", "TL-RL", "RD3P", "ENSEMBLE", "STATUS"])
    map_html = build_tactical_gis_map(theater_key, float(route), final_action, coverage_ratio)

    directive = f"## ◼ DISPATCH DIRECTIVE\n\n# {final_action}"
    telemetry = (
        f"### LOGISTICS TELEMETRY\n\n"
        f"**FORECAST DEMAND**  \n`{demand:,.0f} UNITS`\n\n"
        f"**ON-HAND STOCK**  \n`{float(inventory):,.0f} UNITS`\n\n"
        f"**COVERAGE RATIO**  \n`{coverage_ratio:.2f}`\n\n"
        f"**PROJECTED SHORTAGE**  \n`{shortage:,.0f} UNITS`"
    )
    risk_summary = (
        f"### THREAT / RISK ASSESSMENT\n\n"
        f"**STATUS**  \n`{risk_tier}`\n\n"
        f"**RISK INDEX**  \n`{risk_score:.3f}`\n\n"
        f"**ROUTE INTEGRITY**  \n`{float(route) * 100:.1f}%`\n\n"
        f"**CAPACITY HEADROOM**  \n`{float(capacity) * 100:.1f}%`\n\n"
        f"**SECTOR PRIORITY**  \n`{float(priority) * 100:.1f}%`"
    )
    consensus = (
        f"### POLICY CONSENSUS\n\n"
        f"**TL-RL POLICY**  \n`{tactical_choice}`\n\n"
        f"**RD3P POLICY**  \n`{strategic_choice}`\n\n"
        f"**FINAL CONSENSUS**  \n`{'UNANIMOUS' if tactical_choice == strategic_choice else '50/50 ENSEMBLE'}`"
    )
    status = (
        f"**SYSTEM STATUS:** `{'ONLINE' if allowed else 'DEGRADED'}`   "
        f"**THEATER:** `{theater_key}`   "
        f"**FEASIBLE ACTIONS:** `{len(allowed)}/{len(ACTIONS)}`"
    )

    return directive, telemetry, risk_summary, consensus, score_df, map_html, status
