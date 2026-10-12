"""
EduRisk AI — Explainable Machine Learning for Early Prediction of Student Academic Risk.
Streamlit Web Application for Decision Support, Explainability, and Batch Screening.
"""

import sys
import io
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from app.utils import get_cached_predictor, load_metrics_and_metadata, get_sample_student_profiles
from app.components.risk_card import render_risk_card
from app.components.explainability_view import render_explainability_view
from app.components.benchmark_table import render_benchmark_table

# Page Configuration
st.set_page_config(
    page_title="EduRisk AI — Student Academic Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 25px;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #1a237e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8eaf6 !important;
        color: #1a237e !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<h1 class="main-title">🎓 EduRisk AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Explainable Machine Learning for Early Prediction of Student Academic Risk • 19MAM56 AML Lab Research System</p>',
    unsafe_allow_html=True,
)

# Load System Predictor and Metrics
try:
    predictor = get_cached_predictor()
    metrics_bundle = load_metrics_and_metadata()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}. Please ensure model training was executed.")
    st.stop()

# Sidebar: Controls and Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("System Controls")
    
    st.markdown("### ⚙️ Risk Decision Threshold")
    selected_threshold = st.slider(
        "Sensitivity Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=float(predictor.default_threshold),
        step=0.05,
        help="Lower values prioritize high recall (detecting more at-risk students at the expense of false alarms)."
    )

    st.markdown("---")
    st.markdown("### 📚 Data & Model Provenance")
    st.markdown("**Dataset**: UCI Student Performance (Portuguese Cohort)")
    st.markdown(f"**Total Cohort Size**: {metrics_bundle['metadata']['train_samples'] + metrics_bundle['metadata']['val_samples'] + metrics_bundle['metadata']['test_samples']} students")
    st.markdown(f"**Primary Model**: Tuned XGBoost (Cost-Sensitive)")
    st.markdown(f"**Validation PR-AUC**: {metrics_bundle['optuna_best_params'].get('eval_metric', 'N/A')}")
    st.markdown(f"**Test ROC-AUC**: {metrics_bundle['benchmark_results']['XGBoost_Tuned']['roc_auc']:.3f}")
    st.markdown(f"**Test Minority Recall**: {metrics_bundle['benchmark_results']['XGBoost_Tuned']['recall']:.1%}")

    st.markdown("---")
    st.caption("EduRisk AI is designed strictly as an academic decision support system to aid timely pedagogical interventions.")

# Navigation Tabs
tab_overview, tab_individual, tab_batch, tab_benchmarks, tab_fairness = st.tabs([
    "🏛️ Overview & Analytics",
    "👤 Student Risk Assessment & XAI",
    "👥 Cohort Batch Screening",
    "📈 Benchmarks & Research Visualizations",
    "⚖️ Data Quality, Fairness & Ethics",
])

# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("System Architecture & Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Test Minority Recall", f"{metrics_bundle['benchmark_results']['XGBoost_Tuned']['recall']:.1%}", "At-Risk Detection Rate")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Test ROC-AUC", f"{metrics_bundle['benchmark_results']['XGBoost_Tuned']['roc_auc']:.3f}", "+0.405 vs Baseline")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("PCA Variance Retention", f"{metrics_bundle['pca_experiment']['total_explained_variance']:.1%}", "29 / 63 Components")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Data Completeness", "100.0%", "0 Missing Cells")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    col_intro, col_pipeline = st.columns([1.2, 1])
    with col_intro:
        st.markdown("### 🎯 Problem Statement & Research Objective")
        st.markdown(
            """
            Secondary and higher educational institutions frequently observe academic failure rates that could be
            mitigated through timely intervention. However, institutional alerts commonly trigger only **after**
            midterm examinations, when student trajectories have solidified.
            
            **EduRisk AI** addresses this challenge by formulating a mathematically sound, **early-stage** warning model:
            - **Strict Leakage Prevention**: Drops final grades ($G_3$) and midterm grades ($G_2$), utilizing only demographic indicators, behavioral habits, prior academic friction, and initial diagnostic scores ($G_1$).
            - **Cost-Sensitive Learning**: Mitigates 5.5:1 class imbalance via minority class weighting and threshold tuning, ensuring **high sensitivity (85% recall)** for vulnerable students.
            - **Transparent Explainability**: Unpacks black-box gradient boosting using **TreeSHAP** local attributions and translates statistical risk drivers into concrete educational support actions.
            """
        )
    with col_pipeline:
        st.markdown("### 🛠️ Machine Learning Pipeline Architecture")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A[Raw Educational Data] --> B[Feature Engineering & Domain Ratios]
                B --> C[Stratified Split 65/15/20]
                C --> D[ColumnTransformer Preprocessor]
                D --> E[Imputation & Scaling + OneHot]
                E --> F[Optuna Hyperparameter Tuning]
                F --> G[Cost-Sensitive XGBoost]
                G --> H[TreeSHAP Attribution Engine]
                H --> I[Evidence-Based Pedagogical Actions]
            ```
            """
        )

# =============================================================================
# TAB 2: INDIVIDUAL RISK ASSESSMENT & EXPLAINABILITY
# =============================================================================
with tab_individual:
    st.subheader("👤 Individual Student Risk Assessment & Explainable Diagnostics")
    st.markdown("Select a predefined student archetype or input a specific student profile to evaluate early academic risk.")

    sample_profiles = get_sample_student_profiles()
    selected_archetype = st.selectbox(
        "Load Predefined Student Archetype",
        ["Custom Input"] + list(sample_profiles.keys()),
        index=1,
    )

    if selected_archetype != "Custom Input":
        preset = sample_profiles[selected_archetype]
    else:
        preset = sample_profiles["Profile B: Average / Borderline Student"]

    with st.form("student_assessment_form"):
        st.markdown("#### 1. Demographic & Family Background")
        c1, c2, c3, c4 = st.columns(4)
        school = c1.selectbox("School", ["GP", "MS"], index=0 if preset["school"] == "GP" else 1)
        sex = c2.selectbox("Sex", ["F", "M"], index=0 if preset["sex"] == "F" else 1)
        age = c3.slider("Age", 15, 22, int(preset["age"]))
        address = c4.selectbox("Address", ["U", "R"], index=0 if preset["address"] == "U" else 1, format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)")

        c5, c6, c7, c8 = st.columns(4)
        famsize = c5.selectbox("Family Size", ["GT3", "LE3"], index=0 if preset["famsize"] == "GT3" else 1)
        pstatus = c6.selectbox("Parent Cohabitation", ["T", "A"], index=0 if preset["Pstatus"] == "T" else 1, format_func=lambda x: "Together (T)" if x == "T" else "Apart (A)")
        medu = c7.slider("Mother's Education (0-4)", 0, 4, int(preset["Medu"]))
        fedu = c8.slider("Father's Education (0-4)", 0, 4, int(preset["Fedu"]))

        c9, c10, c11, c12 = st.columns(4)
        mjob = c9.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(preset["Mjob"]))
        fjob = c10.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(preset["Fjob"]))
        reason = c11.selectbox("School Choice Reason", ["home", "reputation", "course", "other"], index=["home", "reputation", "course", "other"].index(preset["reason"]))
        guardian = c12.selectbox("Guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(preset["guardian"]))

        st.markdown("#### 2. Academic Friction & Support Systems")
        a1, a2, a3, a4 = st.columns(4)
        traveltime = a1.slider("Travel Time (1: <15m, 4: >1h)", 1, 4, int(preset["traveltime"]))
        studytime = a2.slider("Weekly Study Time (1-4)", 1, 4, int(preset["studytime"]))
        failures = a3.slider("Past Class Failures (0-4)", 0, 4, int(preset["failures"]))
        g1_score = a4.slider("First Period Diagnostic Score G1 (0-20)", 0, 20, int(preset["G1"]))

        s1, s2, s3, s4 = st.columns(4)
        schoolsup = s1.selectbox("School Support", ["yes", "no"], index=0 if preset["schoolsup"] == "yes" else 1)
        famsup = s2.selectbox("Family Support", ["yes", "no"], index=0 if preset["famsup"] == "yes" else 1)
        paid = s3.selectbox("Paid Extra Classes", ["yes", "no"], index=0 if preset["paid"] == "yes" else 1)
        activities = s4.selectbox("Extracurricular Activities", ["yes", "no"], index=0 if preset["activities"] == "yes" else 1)

        s5, s6, s7 = st.columns(3)
        nursery = s5.selectbox("Attended Nursery", ["yes", "no"], index=0 if preset["nursery"] == "yes" else 1)
        higher = s6.selectbox("Desires Higher Education", ["yes", "no"], index=0 if preset["higher"] == "yes" else 1)
        internet = s7.selectbox("Internet Access at Home", ["yes", "no"], index=0 if preset["internet"] == "yes" else 1)

        st.markdown("#### 3. Behavioral, Social & Lifestyle Factors")
        l1, l2, l3, l4 = st.columns(4)
        famrel = l1.slider("Family Relationship Quality (1-5)", 1, 5, int(preset["famrel"]))
        freetime = l2.slider("Free Time After School (1-5)", 1, 5, int(preset["freetime"]))
        goout = l3.slider("Going Out with Friends (1-5)", 1, 5, int(preset["goout"]))
        absences = l4.slider("School Absences (Days)", 0, 40, int(preset["absences"]))

        b1, b2, b3, b4 = st.columns(4)
        dalc = b1.slider("Workday Alcohol Consumption (1-5)", 1, 5, int(preset["Dalc"]))
        walc = b2.slider("Weekend Alcohol Consumption (1-5)", 1, 5, int(preset["Walc"]))
        health = b3.slider("Health Status (1-5)", 1, 5, int(preset["health"]))
        romantic = b4.selectbox("Romantic Relationship", ["yes", "no"], index=0 if preset["romantic"] == "yes" else 1)

        submitted = st.form_submit_button("🚀 Compute Academic Risk Assessment & XAI Breakdown", use_container_width=True)

    if submitted or selected_archetype != "Custom Input":
        student_data = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": pstatus,
            "Medu": medu,
            "Fedu": fedu,
            "Mjob": mjob,
            "Fjob": fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": dalc,
            "Walc": walc,
            "health": health,
            "absences": absences,
            "G1": g1_score,
        }

        with st.spinner("Executing pipeline inference and generating SHAP local attribution..."):
            result = predictor.predict_student(student_data, threshold=selected_threshold)

        st.markdown("---")
        render_risk_card(result)
        st.markdown("---")
        render_explainability_view(result)

# =============================================================================
# TAB 3: BATCH SCREENING
# =============================================================================
with tab_batch:
    st.subheader("👥 Cohort Batch Screening & Risk Stratification")
    st.markdown("Upload a student roster CSV to generate early warning predictions across an entire cohort.")

    uploaded_file = st.file_uploader("Upload Student Data (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep="," if "," in uploaded_file.getvalue()[:200].decode("utf-8") else ";")
            st.success(f"Successfully loaded {len(batch_df)} student records.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            batch_df = None
    else:
        st.info("No file uploaded. Loading held-out test cohort (N=130) as live demonstration:")
        test_path = PROJECT_ROOT / "data/processed/test.csv"
        batch_df = pd.read_csv(test_path)
        if "at_risk" in batch_df.columns:
            batch_df = batch_df.drop(columns=["at_risk"])

    if batch_df is not None:
        with st.spinner("Processing cohort batch through feature transformer and model pipeline..."):
            scored_df = predictor.predict_batch(batch_df, threshold=selected_threshold)

        total_students = len(scored_df)
        at_risk_students = (scored_df["predicted_academic_risk"] == "At Risk").sum()
        risk_rate = at_risk_students / max(1, total_students)

        b1, b2, b3 = st.columns(3)
        b1.metric("Total Cohort Screened", f"{total_students} students")
        b2.metric("Flagged for Support (At Risk)", f"{at_risk_students} students", f"{risk_rate:.1%} of cohort", delta_color="inverse")
        b3.metric("Operating Cutoff", f"{selected_threshold:.0%}")

        st.markdown("### Risk Distribution Breakdown")
        tier_counts = scored_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Student Count"]
        st.bar_chart(data=tier_counts, x="Risk Tier", y="Student Count")

        st.markdown("### Screened Cohort Results Table")
        st.dataframe(
            scored_df[[
                "predicted_academic_risk",
                "risk_tier",
                "predicted_risk_probability",
                "G1",
                "failures",
                "absences",
                "studytime",
                "age",
                "sex",
                "school",
            ]],
            use_container_width=True,
        )

        # Download CSV export
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Scored Cohort Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="edurisk_cohort_screening_results.csv",
            mime="text/csv",
        )

# =============================================================================
# TAB 4: BENCHMARKS & RESEARCH VISUALIZATIONS
# =============================================================================
with tab_benchmarks:
    st.subheader("📈 Model Benchmarks & Analytical Visualizations")
    render_benchmark_table(metrics_bundle)

    st.markdown("---")
    st.subheader("🎨 Analytical Visualizations Gallery (Reports & Figures)")
    
    figures_dir = PROJECT_ROOT / "reports/figures"
    fig_files = sorted(list(figures_dir.glob("*.png")))

    if fig_files:
        for i in range(0, len(fig_files), 2):
            cols = st.columns(2)
            with cols[0]:
                if i < len(fig_files):
                    img1 = Image.open(fig_files[i])
                    st.image(img1, caption=fig_files[i].name, use_column_width=True)
            with cols[1]:
                if i + 1 < len(fig_files):
                    img2 = Image.open(fig_files[i + 1])
                    st.image(img2, caption=fig_files[i + 1].name, use_column_width=True)
    else:
        st.warning("No generated figures found in reports/figures/. Run evaluate.py first.")

# =============================================================================
# TAB 5: FAIRNESS & ETHICAL GOVERNANCE
# =============================================================================
with tab_fairness:
    st.subheader("⚖️ Data Quality, Algorithmic Fairness & Ethical Governance")
    
    st.markdown("### 📊 Subgroup Performance Disparity Analysis")
    st.markdown(
        "Responsible machine learning requires evaluating performance parity across demographic subgroups "
        "to ensure the early warning system does not disproportionately overlook or over-flag protected groups."
    )

    fairness_report_path = PROJECT_ROOT / "reports/subgroup_fairness_report.json"
    if fairness_report_path.exists():
        fairness_df = pd.read_json(fairness_report_path)
        st.dataframe(
            fairness_df.style.format({
                "recall": "{:.1%}",
                "precision": "{:.1%}",
                "false_positive_rate": "{:.1%}",
            }),
            use_container_width=True,
        )

    st.markdown("### 🛡️ Ethical Principles for Academic Decision Support")
    st.markdown(
        """
        1. **Supportive, Never Punitive**: EduRisk AI is exclusively designed to direct proactive tutoring, counseling, and faculty outreach. It must **never** be utilized for negative academic tracking, disciplinary actions, or gatekeeping educational opportunities.
        2. **Human-in-the-Loop Authority**: Model predictions represent probabilistic estimates derived from historical cohorts. Final pedagogical decisions remain strictly under human faculty and advisory authority.
        3. **Preventing Self-Fulfilling Prophecies**: Risk predictions must not be disclosed in pejorative terminology to students. Communications should focus on growth-mindset resource offers.
        4. **Data Privacy & Epistemic Boundaries**: No identifiable student names or private IDs are stored or transmitted. The model assesses academic patterns within documented feature limits.
        """
    )
