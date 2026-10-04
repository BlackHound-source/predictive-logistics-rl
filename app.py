"""Main application entry point and Gradio frontend configuration."""

import os
import pathlib
import pandas as pd
import gradio as gr

from config import THEATERS
from agents.trainer import train_agents
from core.decision_engine import evaluate_decision

# 1. Warm-up pretraining
tl_agent, rd3p_agent = train_agents()

# 2. Load tactical stylesheet
css_path = pathlib.Path(__file__).parent / "assets" / "tactical.css"
with open(css_path, "r", encoding="utf-8") as f:
    tactical_css = f.read()

# 3. Decision wrapper
def handle_execution(*args):
    return evaluate_decision(tl_agent, rd3p_agent, *args)

# 4. Gradio UI Layout
with gr.Blocks(
    css=tactical_css,
    title="Tactical Predictive Logistics Command",
    theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
) as app:

    gr.HTML(
        """
        <div class="command-header">
            <div class="command-title">◈ TACTICAL PREDICTIVE LOGISTICS MANAGEMENT SYSTEM</div>
            <div class="command-subtitle">
                AUTONOMOUS LOGISTICS DECISION SUPPORT // TL-RL + RD3P CONSENSUS // PREDICTIVE DEMAND INTELLIGENCE
            </div>
        </div>
        """
    )

    status_display = gr.Markdown(
        "**SYSTEM STATUS:** `STANDBY` &nbsp;&nbsp; **DECISION ENGINE:** `READY` &nbsp;&nbsp; **RL POLICY:** `LOADED` &nbsp;&nbsp; **GIS:** `ONLINE`",
        elem_classes=["status-bar"],
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["command-panel"]):
            gr.Markdown("### ▣ OPERATIONAL TELEMETRY INPUT", elem_classes=["section-header"])
            in_theater = gr.Dropdown(choices=list(THEATERS.keys()), value="LADAKH_NORTHERN_AXIS", label="THEATER / SECTOR")
            in_month = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="FORECAST HORIZON / MONTH")
            in_inv = gr.Number(value=11000, label="CURRENT STOCK / UNITS")
            in_surge = gr.Slider(minimum=0.8, maximum=2.0, value=1.0, step=0.1, label="OPERATIONAL TEMPO / SURGE")
            in_weather = gr.Slider(minimum=0.0, maximum=1.0, value=0.30, step=0.05, label="WEATHER / TERRAIN FRICTION")
            in_route = gr.Slider(minimum=0.0, maximum=1.0, value=0.90, step=0.05, label="PRIMARY ROUTE INTEGRITY")
            in_cap = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.05, label="FLEET CAPACITY HEADROOM")
            in_trend = gr.Slider(minimum=-0.20, maximum=0.50, value=0.05, step=0.05, label="DEMAND ESCALATION TREND")
            in_pri = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="SECTOR STRATEGIC PRIORITY")
            btn_execute = gr.Button("▶ EXECUTE DECISION MATRIX", variant="primary", elem_classes=["execute-button"])

        with gr.Column(scale=2):
            out_directive = gr.Markdown("## ◼ DISPATCH DIRECTIVE\n\n# AWAITING TELEMETRY", elem_classes=["directive-panel"])
            with gr.Row():
                out_telemetry = gr.Markdown("### LOGISTICS TELEMETRY\n\nAwaiting decision execution.", elem_classes=["telemetry-card"])
                out_risk = gr.Markdown("### THREAT / RISK ASSESSMENT\n\nSystem awaiting telemetry.", elem_classes=["risk-card"])
                out_consensus = gr.Markdown("### POLICY CONSENSUS\n\nTL-RL: STANDBY\nRD3P: STANDBY", elem_classes=["consensus-card"])

            gr.Markdown("### ▣ DECISION MATRIX / FEASIBILITY MASK", elem_classes=["section-header"])
            out_scores = gr.Dataframe(
                headers=["ACTION", "TL-RL", "RD3P", "ENSEMBLE", "STATUS"],
                datatype=["str", "number", "number", "number", "str"],
                value=pd.DataFrame(columns=["ACTION", "TL-RL", "RD3P", "ENSEMBLE", "STATUS"]),
                interactive=False,
                elem_classes=["command-panel"],
            )

    gr.Markdown("### ▣ SITUATIONAL CORRIDOR / NETWORK OVERVIEW", elem_classes=["section-header"])
    with gr.Column(elem_classes=["map-panel"]):
        out_map = gr.HTML(
            '<div style="height:520px;display:flex;align-items:center;justify-content:center;background:#080b08;color:#596454;font-family:monospace;letter-spacing:2px;border:1px solid #293329;">GIS SYSTEM STANDBY</div>'
        )

    gr.HTML('<div class="command-footer">TACTICAL PREDICTIVE LOGISTICS SYSTEM // DECISION SUPPORT FRAMEWORK</div>')

    btn_execute.click(
        fn=handle_execution,
        inputs=[in_theater, in_month, in_inv, in_surge, in_weather, in_route, in_cap, in_trend, in_pri],
        outputs=[out_directive, out_telemetry, out_risk, out_consensus, out_scores, out_map, status_display],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
