"""
Risk Assessment Card Component for EduRisk AI Streamlit Dashboard.
"""

import streamlit as st
from typing import Dict, Any


def render_risk_card(prediction_result: Dict[str, Any]):
    """Renders visual risk alert banner and metric cards."""
    proba = prediction_result["risk_probability"]
    status = prediction_result["risk_status"]
    tier = prediction_result["risk_level"]
    th = prediction_result["applied_threshold"]

    if status == "At Risk":
        color = "#c62828"
        bg = "#ffebee"
        border = "#ef9a9a"
        icon = "⚠️"
    else:
        color = "#2e7d32"
        bg = "#e8f5e9"
        border = "#a5d6a7"
        icon = "✅"

    st.markdown(
        f"""
        <div style="background-color: {bg}; border: 2px solid {border}; border-radius: 10px; padding: 18px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="color: {color}; margin: 0; font-size: 1.6rem;">{icon} {status} — <span style="font-weight: 500;">{tier}</span></h2>
                    <p style="color: #444; margin: 6px 0 0 0; font-size: 0.95rem;">{prediction_result["confidence_assessment"]}</p>
                </div>
                <div style="text-align: right; background: white; padding: 10px 18px; border-radius: 8px; border: 1px solid {border};">
                    <span style="font-size: 0.85rem; color: #666; text-transform: uppercase; font-weight: bold;">Risk Score</span><br/>
                    <span style="font-size: 2.1rem; font-weight: bold; color: {color};">{proba:.1%}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Academic Status", status)
    c2.metric("Probability of Risk", f"{proba:.2%}", delta=f"{proba - th:+.1%} vs cutoff" if proba >= th else f"{proba - th:.1%} vs cutoff", delta_color="inverse")
    c3.metric("Decision Threshold Cutoff", f"{th:.2%}", help="Configured sensitivity cutoff for flag triggering")

    st.info(f"ℹ️ **Decision Support Protocol**: {prediction_result['epistemic_disclaimer']}")
