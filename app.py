"""
EduRisk AI — Explainable Machine Learning for Early Prediction of Student Academic Risk
Hugging Face Spaces Gradio Application.
"""

import sys
import json
from pathlib import Path
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.models.predict import EduRiskPredictor

# Initialize Predictor singleton
predictor = EduRiskPredictor()
config = load_config()

with open(PROJECT_ROOT / config["paths"]["metrics_path"], "r", encoding="utf-8") as f:
    metrics_bundle = json.load(f)

ARCHETYPES = {
    "Profile A: Student in Need of Urgent Academic Support": {
        "school": "GP", "sex": "M", "age": 18, "address": "R", "famsize": "GT3", "Pstatus": "A",
        "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "other", "reason": "course", "guardian": "other",
        "traveltime": 3, "studytime": 1, "failures": 2, "schoolsup": "no", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "no", "higher": "no", "internet": "no", "romantic": "yes",
        "famrel": 2, "freetime": 4, "goout": 4, "Dalc": 3, "Walc": 4, "health": 3, "absences": 12, "G1": 7,
    },
    "Profile B: Average / Borderline Student": {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 2, "Mjob": "services", "Fjob": "services", "reason": "home", "guardian": "mother",
        "traveltime": 2, "studytime": 2, "failures": 0, "schoolsup": "yes", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4, "absences": 4, "G1": 10,
    },
    "Profile C: Resilient / High-Performing Student": {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "LE3", "Pstatus": "T",
        "Medu": 4, "Fedu": 4, "Mjob": "health", "Fjob": "teacher", "reason": "reputation", "guardian": "mother",
        "traveltime": 1, "studytime": 3, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "yes",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 5, "freetime": 2, "goout": 2, "Dalc": 1, "Walc": 1, "health": 5, "absences": 1, "G1": 15,
    },
}


def predict_single(
    school, sex, age, address, famsize, pstatus, medu, fedu, mjob, fjob, reason, guardian,
    traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
    famrel, freetime, goout, dalc, walc, health, absences, g1, threshold
):
    student_dict = {
        "school": school, "sex": sex, "age": int(age), "address": address, "famsize": famsize, "Pstatus": pstatus,
        "Medu": int(medu), "Fedu": int(fedu), "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
        "traveltime": int(traveltime), "studytime": int(studytime), "failures": int(failures),
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid, "activities": activities,
        "nursery": nursery, "higher": higher, "internet": internet, "romantic": romantic,
        "famrel": int(famrel), "freetime": int(freetime), "goout": int(goout),
        "Dalc": int(dalc), "Walc": int(walc), "health": int(health), "absences": int(absences), "G1": int(g1),
    }

    result = predictor.predict_student(student_dict, threshold=float(threshold))
    proba = result["risk_probability"]
    status_label = result["risk_status"]
    tier = result["risk_level"]
    th = result["applied_threshold"]

    # Status banner markdown
    if status_label == "At Risk":
        color = "#d32f2f"
        icon = "⚠️"
        bg = "#ffebee"
    else:
        color = "#2e7d32"
        icon = "✅"
        bg = "#e8f5e9"

    summary_html = f"""
    <div style="background-color: {bg}; border: 2px solid {color}; border-radius: 8px; padding: 15px; margin-bottom: 12px;">
        <h2 style="color: {color}; margin: 0 0 8px 0; font-size: 1.5rem;">{icon} {status_label} — {tier}</h2>
        <p style="margin: 0; font-size: 1rem; color: #333;">
            <b>Predicted Risk Probability:</b> <span style="font-size: 1.4rem; color: {color}; font-weight: bold;">{proba:.1%}</span>
            &nbsp;|&nbsp; <b>Alert Cutoff Threshold:</b> {th:.1%}
        </p>
        <p style="margin: 8px 0 0 0; font-size: 0.9rem; color: #555;">{result['confidence_assessment']}</p>
    </div>
    """

    # Local SHAP Plot
    drivers = result.get("top_risk_drivers", [])
    protective = result.get("top_protective_factors", [])
    combined = []
    for d in drivers:
        combined.append({"feature": d["feature"], "shap": d["shap_value"], "color": "#d32f2f"})
    for p in protective:
        combined.append({"feature": p["feature"], "shap": p["shap_value"], "color": "#388e3c"})

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    if combined:
        plot_df = pd.DataFrame(combined).sort_values("shap", ascending=True)
        ax.barh(plot_df["feature"], plot_df["shap"], color=plot_df["color"], edgecolor="black", height=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP Impact on Risk Log-Odds", fontsize=9)
        ax.set_title("Local Feature Attribution (Red: Increases Risk, Green: Reduces Risk)", fontsize=10)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "Balanced Risk Indicators", ha="center", va="center")
    plt.tight_layout()

    # Interventions markdown
    interventions = result.get("evidence_based_interventions", [])
    interventions_md = "### 📋 Recommended Pedagogical Interventions:\n"
    for idx, rec in enumerate(interventions, 1):
        interventions_md += f"**{idx}.** {rec}\n\n"

    disclaimer_md = f"> ℹ️ **Epistemic Disclaimer**: {result['epistemic_disclaimer']}"

    return summary_html, fig, interventions_md, disclaimer_md


def load_preset(archetype_name):
    data = ARCHETYPES[archetype_name]
    return (
        data["school"], data["sex"], data["age"], data["address"], data["famsize"], data["Pstatus"],
        data["Medu"], data["Fedu"], data["Mjob"], data["Fjob"], data["reason"], data["guardian"],
        data["traveltime"], data["studytime"], data["failures"], data["schoolsup"], data["famsup"],
        data["paid"], data["activities"], data["nursery"], data["higher"], data["internet"], data["romantic"],
        data["famrel"], data["freetime"], data["goout"], data["Dalc"], data["Walc"], data["health"],
        data["absences"], data["G1"]
    )


def process_batch(file_obj, threshold):
    if file_obj is None:
        test_path = PROJECT_ROOT / "data/processed/test.csv"
        df = pd.read_csv(test_path)
        if "at_risk" in df.columns:
            df = df.drop(columns=["at_risk"])
    else:
        df = pd.read_csv(file_obj.name, sep="," if "," in open(file_obj.name, "r").read(200) else ";")

    scored_df = predictor.predict_batch(df, threshold=float(threshold))
    total = len(scored_df)
    at_risk_count = int((scored_df["predicted_academic_risk"] == "At Risk").sum())
    risk_rate = at_risk_count / max(1, total)

    summary_text = (
        f"**Total Students Screened:** {total} | "
        f"**Flagged for Support (At-Risk):** {at_risk_count} ({risk_rate:.1%}) | "
        f"**Decision Threshold:** {float(threshold):.1%}"
    )

    out_csv = PROJECT_ROOT / "reports" / "batch_prediction_output.csv"
    scored_df.to_csv(out_csv, index=False)

    display_cols = [
        "predicted_academic_risk", "risk_tier", "predicted_risk_probability",
        "G1", "failures", "absences", "studytime", "age", "sex", "school"
    ]
    display_df = scored_df[[c for c in display_cols if c in scored_df.columns]]

    return summary_text, display_df, str(out_csv)


# Build Gradio UI
with gr.Blocks(title="EduRisk AI — Student Academic Risk Prediction", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🎓 EduRisk AI: Early Prediction of Student Academic Risk
        ### Explainable Machine Learning Decision Support System • 19MAM56 AML Lab Research Project
        ---
        """
    )

    with gr.Tabs():
        # TAB 1: INDIVIDUAL RISK ASSESSMENT
        with gr.TabItem("👤 Individual Student Assessment & XAI"):
            gr.Markdown("### Load Predefined Student Archetype or Configure Inputs Below:")
            with gr.Row():
                preset_selector = gr.Dropdown(
                    choices=list(ARCHETYPES.keys()),
                    value="Profile A: Student in Need of Urgent Academic Support",
                    label="Quick-Load Student Archetype",
                )
                threshold_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.35, step=0.05, label="Decision Threshold Cutoff"
                )

            with gr.Accordion("1. Student Demographics & Family Background", open=True):
                with gr.Row():
                    school = gr.Dropdown(["GP", "MS"], value="GP", label="School")
                    sex = gr.Dropdown(["M", "F"], value="M", label="Sex")
                    age = gr.Slider(15, 22, value=18, step=1, label="Age")
                    address = gr.Dropdown(["U", "R"], value="R", label="Address (U=Urban, R=Rural)")
                with gr.Row():
                    famsize = gr.Dropdown(["GT3", "LE3"], value="GT3", label="Family Size")
                    pstatus = gr.Dropdown(["T", "A"], value="A", label="Parent Cohabitation (T=Together, A=Apart)")
                    medu = gr.Slider(0, 4, value=1, step=1, label="Mother's Education (0-4)")
                    fedu = gr.Slider(0, 4, value=1, step=1, label="Father's Education (0-4)")
                with gr.Row():
                    mjob = gr.Dropdown(["teacher", "health", "services", "at_home", "other"], value="at_home", label="Mother's Job")
                    fjob = gr.Dropdown(["teacher", "health", "services", "at_home", "other"], value="other", label="Father's Job")
                    reason = gr.Dropdown(["home", "reputation", "course", "other"], value="course", label="Reason for School")
                    guardian = gr.Dropdown(["mother", "father", "other"], value="other", label="Guardian")

            with gr.Accordion("2. Academic Friction & Educational Support", open=True):
                with gr.Row():
                    traveltime = gr.Slider(1, 4, value=3, step=1, label="Travel Time (1: <15m, 4: >1h)")
                    studytime = gr.Slider(1, 4, value=1, step=1, label="Study Time (1: <2h, 4: >10h)")
                    failures = gr.Slider(0, 4, value=2, step=1, label="Prior Class Failures (0-4)")
                    g1 = gr.Slider(0, 20, value=7, step=1, label="Early Diagnostic Grade G1 (0-20)")
                with gr.Row():
                    schoolsup = gr.Dropdown(["yes", "no"], value="no", label="Extra School Support")
                    famsup = gr.Dropdown(["yes", "no"], value="no", label="Family Support")
                    paid = gr.Dropdown(["yes", "no"], value="no", label="Paid Extra Classes")
                    activities = gr.Dropdown(["yes", "no"], value="no", label="Extracurriculars")
                with gr.Row():
                    nursery = gr.Dropdown(["yes", "no"], value="no", label="Attended Nursery")
                    higher = gr.Dropdown(["yes", "no"], value="no", label="Desires Higher Education")
                    internet = gr.Dropdown(["yes", "no"], value="no", label="Home Internet")
                    romantic = gr.Dropdown(["yes", "no"], value="yes", label="In Romantic Relationship")

            with gr.Accordion("3. Lifestyle, Social & Behavioral Factors", open=True):
                with gr.Row():
                    famrel = gr.Slider(1, 5, value=2, step=1, label="Family Relationship Quality (1-5)")
                    freetime = gr.Slider(1, 5, value=4, step=1, label="Free Time (1-5)")
                    goout = gr.Slider(1, 5, value=4, step=1, label="Going Out with Friends (1-5)")
                    absences = gr.Slider(0, 40, value=12, step=1, label="Absences (Days)")
                with gr.Row():
                    dalc = gr.Slider(1, 5, value=3, step=1, label="Workday Alcohol (1-5)")
                    walc = gr.Slider(1, 5, value=4, step=1, label="Weekend Alcohol (1-5)")
                    health = gr.Slider(1, 5, value=3, step=1, label="Health Status (1-5)")

            predict_btn = gr.Button("🚀 Predict Academic Risk & Generate SHAP Diagnostics", variant="primary")

            # Outputs
            result_html = gr.HTML(label="Risk Assessment Summary")
            with gr.Row():
                shap_plot = gr.Plot(label="Local SHAP Attribution Breakdown")
                with gr.Column():
                    interventions_output = gr.Markdown()
                    disclaimer_output = gr.Markdown()

            # Wire Archetype Selection
            all_inputs = [
                school, sex, age, address, famsize, pstatus, medu, fedu, mjob, fjob, reason, guardian,
                traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher,
                internet, romantic, famrel, freetime, goout, dalc, walc, health, absences, g1
            ]

            preset_selector.change(
                fn=load_preset,
                inputs=[preset_selector],
                outputs=all_inputs
            )

            predict_btn.click(
                fn=predict_single,
                inputs=all_inputs + [threshold_slider],
                outputs=[result_html, shap_plot, interventions_output, disclaimer_output]
            )

        # TAB 2: BATCH SCREENING
        with gr.TabItem("👥 Cohort Batch Screening"):
            gr.Markdown("### Upload a Student Roster (CSV) or Screen the Benchmark Cohort:")
            with gr.Row():
                file_upload = gr.File(label="Upload Student CSV File", file_types=[".csv"])
                batch_threshold = gr.Slider(0.10, 0.90, value=0.35, step=0.05, label="Risk Cutoff")

            batch_btn = gr.Button("📊 Run Batch Screening", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.DataFrame(label="Screened Cohort Predictions")
            batch_download = gr.File(label="Download Scored Results (CSV)")

            batch_btn.click(
                fn=process_batch,
                inputs=[file_upload, batch_threshold],
                outputs=[batch_summary, batch_table, batch_download]
            )

        # TAB 3: BENCHMARK & VISUALIZATIONS
        with gr.TabItem("📈 Model Benchmark & Figures"):
            gr.Markdown("### Empirical Benchmark Comparison on Held-Out Test Set (N=130)")
            bench_df = pd.DataFrame(metrics_bundle["benchmark_results"]).T.reset_index()
            bench_df.rename(columns={"index": "Model Architecture"}, inplace=True)
            gr.DataFrame(bench_df)

            gr.Markdown("### Research Figures Gallery")
            with gr.Row():
                gr.Image(str(PROJECT_ROOT / "reports/figures/05_model_benchmark_comparison.png"), label="Model Benchmark Comparison")
                gr.Image(str(PROJECT_ROOT / "reports/figures/06_confusion_matrix_selected_model.png"), label="Confusion Matrix")
            with gr.Row():
                gr.Image(str(PROJECT_ROOT / "reports/figures/07_roc_and_pr_curves.png"), label="ROC and PR Curves")
                gr.Image(str(PROJECT_ROOT / "reports/figures/shap_importance_bar.png"), label="Top 15 Global SHAP Predictors")
            with gr.Row():
                gr.Image(str(PROJECT_ROOT / "reports/figures/04_pca_explained_variance.png"), label="PCA Explained Variance")
                gr.Image(str(PROJECT_ROOT / "reports/figures/10_subgroup_fairness_analysis.png"), label="Subgroup Demographic Parity")

        # TAB 4: ETHICS & FAIRNESS
        with gr.TabItem("⚖️ Ethical Governance & Limitations"):
            gr.Markdown(
                """
                ### 🛡️ Responsible AI Principles for Academic Decision Support
                1. **Supportive, Never Punitive**: EduRisk AI is exclusively engineered to direct proactive tutoring, counseling, and faculty outreach. It must **never** be used for negative academic tracking, disciplinary actions, or denying opportunities.
                2. **Human-in-the-Loop Sovereign Authority**: Model predictions represent probabilistic estimates derived from historical cohorts. Final pedagogical decisions remain strictly under human faculty authority.
                3. **Association vs. Causality**: Model features reflect statistical associations in observational data, not deterministic causal laws.
                4. **Data Privacy**: No identifiable student PII (names, student IDs) is processed or transmitted.
                """
            )

    pass

if __name__ == "__main__":
    demo.launch()
