"""
Explainability and Academic Intervention Panel for Streamlit.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List


def render_explainability_view(prediction_result: Dict[str, Any]):
    """Renders local feature attributions and actionable intervention advice."""
    st.subheader("🔍 Explainable AI Diagnostics (Local SHAP Attribution)")
    st.markdown(
        "Below is a mathematical breakdown of the individual factors that pushed the risk probability "
        "**higher** (risk drivers) versus **lower** (protective factors)."
    )

    drivers = prediction_result.get("top_risk_drivers", [])
    protective = prediction_result.get("top_protective_factors", [])

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🔴 Key Risk Factors (Pushed Towards At-Risk)")
        if drivers:
            driver_df = pd.DataFrame(drivers)[["feature", "shap_value", "abs_impact"]]
            driver_df.columns = ["Factor", "SHAP Impact (+)", "Magnitude"]
            st.dataframe(
                driver_df.style.format({"SHAP Impact (+)": "+{:.3f}", "Magnitude": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No major risk exacerbating factors identified.")

    with col_right:
        st.markdown("### 🟢 Protective Factors (Pushed Towards Success)")
        if protective:
            prot_df = pd.DataFrame(protective)[["feature", "shap_value", "abs_impact"]]
            prot_df.columns = ["Factor", "SHAP Impact (-)", "Magnitude"]
            st.dataframe(
                prot_df.style.format({"SHAP Impact (-)": "{:.3f}", "Magnitude": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No protective factors identified.")

    # Visualization of feature impact bar
    combined = []
    for d in drivers:
        combined.append({"feature": d["feature"], "shap": d["shap_value"], "color": "#d32f2f"})
    for p in protective:
        combined.append({"feature": p["feature"], "shap": p["shap_value"], "color": "#388e3c"})

    if combined:
        plot_df = pd.DataFrame(combined).sort_values("shap", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
        ax.barh(plot_df["feature"], plot_df["shap"], color=plot_df["color"], edgecolor="black", height=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP Contribution to Risk Log-Odds", fontsize=9)
        ax.set_title("Local Feature Attribution (Red = Increases Risk, Green = Reduces Risk)", fontsize=10)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Evidence-based Actionable Interventions
    st.subheader("📋 Recommended Pedagogical & Institutional Interventions")
    interventions = prediction_result.get("evidence_based_interventions", [])
    if interventions:
        for idx, rec in enumerate(interventions, 1):
            st.markdown(f"**{idx}.** {rec}")
    else:
        st.write("Maintain current educational tracking.")
