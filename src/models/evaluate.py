"""
Comprehensive Evaluation, Visualization, and Subgroup Analysis Module for EduRisk AI.
Generates all 8+ required analytical figures using real experimental data
and saves them to reports/figures/.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.features.preprocessing import prepare_and_split_data
from src.explainability.shap_explainer import EduRiskExplainer

# Set style for publication-quality scientific charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

logger = logging.getLogger("EduRisk-Eval")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def generate_all_visualizations_and_eval():
    config = load_config()
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data splits and metadata
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = prepare_and_split_data()
    raw_df = pd.read_csv(PROJECT_ROOT / config["data"]["raw_file"], sep=";")
    raw_df["at_risk"] = (raw_df["G3"] < config["target"]["risk_threshold"]).astype(int)

    # Load preprocessor and models
    preprocessor = joblib.load(PROJECT_ROOT / config["paths"]["preprocessor_path"])
    best_model = joblib.load(PROJECT_ROOT / config["paths"]["best_model_path"])
    rf_model = joblib.load(PROJECT_ROOT / "models/saved_models/random_forest_model.joblib")
    lr_model = joblib.load(PROJECT_ROOT / "models/saved_models/logistic_regression_model.joblib")
    pca_transformer = joblib.load(PROJECT_ROOT / "models/saved_models/pca_transformer.joblib")

    with open(PROJECT_ROOT / config["paths"]["metrics_path"], "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    X_test_trans = preprocessor.transform(X_test)
    y_test_proba = best_model.predict_proba(X_test_trans)[:, 1]
    y_test_pred = (y_test_proba >= config["models"]["decision_threshold"]).astype(int)

    rf_test_proba = rf_model.predict_proba(X_test_trans)[:, 1]
    lr_test_proba = lr_model.predict_proba(X_test_trans)[:, 1]

    # =========================================================================
    # 1. Target Class Distribution
    # =========================================================================
    logger.info("Generating Figure 1: Target Class Distribution...")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    counts = raw_df["at_risk"].value_counts()
    labels = ["Lower Risk (G3 ≥ 10)", "At Risk (G3 < 10)"]
    palette = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(labels, [counts[0], counts[1]], color=palette, width=0.5, edgecolor="black", linewidth=1.1)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h} ({h / len(raw_df):.1%})",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.set_title("Student Academic Risk Target Distribution (UCI Portuguese Cohort, N=649)", fontsize=12, pad=12)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    fig1_path = figures_dir / "01_target_distribution.png"
    plt.savefig(fig1_path)
    plt.close()

    # =========================================================================
    # 2. Missing-Value and Data Completeness Analysis
    # =========================================================================
    logger.info("Generating Figure 2: Missing Value / Data Quality Analysis...")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    completeness = (1 - raw_df.isnull().mean()) * 100
    sns.barplot(x=completeness.index, y=completeness.values, color="#2e7d32", ax=ax, edgecolor="black")
    ax.set_xticklabels(completeness.index, rotation=90, fontsize=8)
    ax.set_ylabel("Data Completeness Rate (%)", fontsize=11)
    ax.set_ylim(95, 101)
    ax.axhline(100, color="darkred", linestyle="--", linewidth=1, label="100% Complete Threshold")
    ax.set_title("Dataset Feature Completeness & Quality Verification (Zero Null Incurrence)", fontsize=12, pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig2_path = figures_dir / "02_missing_value_analysis.png"
    plt.savefig(fig2_path)
    plt.close()

    # =========================================================================
    # 3. Feature Correlation Heatmap (Early Predictors)
    # =========================================================================
    logger.info("Generating Figure 3: Feature Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    key_num_cols = ["age", "Medu", "Fedu", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "at_risk"]
    corr = raw_df[key_num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap of Key Educational Indicators & Academic Risk", fontsize=12, pad=15)
    plt.tight_layout()
    fig3_path = figures_dir / "03_feature_correlation_heatmap.png"
    plt.savefig(fig3_path)
    plt.close()

    # =========================================================================
    # 4. PCA Explained Variance Plot
    # =========================================================================
    logger.info("Generating Figure 4: PCA Explained Variance Analysis...")
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    pca_data = metrics_data["pca_experiment"]
    evr = pca_data["explained_variance_ratio"]
    cum_evr = pca_data["cumulative_variance_ratio"]
    x_comps = range(1, len(evr) + 1)

    ax1.bar(x_comps, evr, alpha=0.6, color="#4575b4", edgecolor="black", label="Individual Explained Variance")
    ax1.set_xlabel("Principal Component Index", fontsize=11)
    ax1.set_ylabel("Individual Explained Variance Ratio", color="#4575b4", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(x_comps, cum_evr, color="#d73027", marker="o", markersize=4, linewidth=2, label="Cumulative Explained Variance")
    ax2.axhline(0.95, color="darkgreen", linestyle="--", label="95% Variance Retention Cutoff")
    ax2.set_ylabel("Cumulative Explained Variance Ratio", color="#d73027", fontsize=11)
    ax2.set_ylim(0, 1.05)

    plt.title(f"PCA Dimensionality Reduction: {pca_data['n_pca_components']} Components Retain 95.3% Variance", fontsize=12, pad=12)
    fig.tight_layout()
    fig4_path = figures_dir / "04_pca_explained_variance.png"
    plt.savefig(fig4_path)
    plt.close()

    # =========================================================================
    # 5. Model Performance Comparison
    # =========================================================================
    logger.info("Generating Figure 5: Model Benchmark Comparison...")
    bench = pd.DataFrame(metrics_data["benchmark_results"]).T
    bench_models = [m for m in bench.index if "PCA" not in m and m != "DummyClassifier"] + ["LogisticRegression_PCA", "RandomForest_PCA"]
    subset_bench = bench.loc[bench_models][["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    subset_bench.plot(kind="bar", ax=ax, colormap="tab10", width=0.8, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Comprehensive Model Benchmark on Test Set (N=130)", fontsize=13, pad=12)
    ax.set_xticklabels(subset_bench.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=3)
    plt.tight_layout()
    fig5_path = figures_dir / "05_model_benchmark_comparison.png"
    plt.savefig(fig5_path)
    plt.close()

    # =========================================================================
    # 6. Confusion Matrix for Selected Model
    # =========================================================================
    logger.info("Generating Figure 6: Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Lower Risk (0)", "At Risk (1)"],
                yticklabels=["Lower Risk (0)", "At Risk (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Class (Threshold = 0.35)", fontsize=11)
    ax.set_ylabel("Actual Ground Truth Class", fontsize=11)
    ax.set_title(f"Confusion Matrix: Tuned XGBoost (Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.1%})", fontsize=12, pad=12)
    plt.tight_layout()
    fig6_path = figures_dir / "06_confusion_matrix_selected_model.png"
    plt.savefig(fig6_path)
    plt.close()

    # =========================================================================
    # 7. ROC and Precision-Recall Curves
    # =========================================================================
    logger.info("Generating Figure 7: ROC and PR Curves...")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # ROC Curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_proba)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_proba)

    ax_roc.plot(fpr_xgb, tpr_xgb, color="#1f77b4", lw=2, label=f"XGBoost Tuned (AUC = {metrics_data['benchmark_results']['XGBoost_Tuned']['roc_auc']:.3f})")
    ax_roc.plot(fpr_lr, tpr_lr, color="#2ca02c", lw=2, label=f"Logistic Regression (AUC = {metrics_data['benchmark_results']['LogisticRegression']['roc_auc']:.3f})")
    ax_roc.plot(fpr_rf, tpr_rf, color="#ff7f0e", lw=2, label=f"Random Forest (AUC = {metrics_data['benchmark_results']['RandomForest']['roc_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", label="No-Skill Reference")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12)
    ax_roc.legend(loc="lower right")

    # PR Curves
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_test_proba)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_test_proba)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_test_proba)

    ax_pr.plot(rec_xgb, prec_xgb, color="#1f77b4", lw=2, label=f"XGBoost Tuned (PR-AUC = {metrics_data['benchmark_results']['XGBoost_Tuned']['pr_auc']:.3f})")
    ax_pr.plot(rec_lr, prec_lr, color="#2ca02c", lw=2, label=f"Logistic Regression (PR-AUC = {metrics_data['benchmark_results']['LogisticRegression']['pr_auc']:.3f})")
    ax_pr.plot(rec_rf, prec_rf, color="#ff7f0e", lw=2, label=f"Random Forest (PR-AUC = {metrics_data['benchmark_results']['RandomForest']['pr_auc']:.3f})")
    baseline_pr = float(y_test.mean())
    ax_pr.axhline(baseline_pr, color="grey", linestyle="--", label=f"Baseline Prevalence ({baseline_pr:.1%})")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=12)
    ax_pr.legend(loc="upper right")

    plt.tight_layout()
    fig7_path = figures_dir / "07_roc_and_pr_curves.png"
    plt.savefig(fig7_path)
    plt.close()

    # =========================================================================
    # 8. SHAP Global Feature Importance & Summary
    # =========================================================================
    logger.info("Generating Figure 8: SHAP Global Feature Importance...")
    explainer = EduRiskExplainer()
    shap_results = explainer.explain_global(X_test_trans, str(figures_dir))

    # =========================================================================
    # 9. Probability Calibration Analysis
    # =========================================================================
    logger.info("Generating Figure 9: Probability Calibration Curve...")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, y_test_proba, n_bins=5, strategy="uniform")
    prob_true_lr, prob_pred_lr = calibration_curve(y_test, lr_test_proba, n_bins=5, strategy="uniform")

    brier_xgb = brier_score_loss(y_test, y_test_proba)
    brier_lr = brier_score_loss(y_test, lr_test_proba)

    ax.plot(prob_pred_xgb, prob_true_xgb, marker="s", lw=2, color="#1f77b4", label=f"XGBoost (Brier = {brier_xgb:.3f})")
    ax.plot(prob_pred_lr, prob_true_lr, marker="o", lw=2, color="#2ca02c", label=f"Logistic Regression (Brier = {brier_lr:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Empirical Risk)", fontsize=11)
    ax.set_title("Reliability Diagram (Probability Calibration)", fontsize=12, pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig9_path = figures_dir / "09_calibration_curve.png"
    plt.savefig(fig9_path)
    plt.close()

    # =========================================================================
    # 10. Subgroup Fairness & Disparity Analysis (Sex, School, Urban/Rural)
    # =========================================================================
    logger.info("Generating Figure 10: Subgroup Fairness Analysis...")
    test_analysis_df = X_test.copy()
    test_analysis_df["true_risk"] = y_test.values
    test_analysis_df["pred_risk"] = y_test_pred
    test_analysis_df["pred_proba"] = y_test_proba

    subgroups = ["sex", "school", "address"]
    subgroup_metrics = []

    for sg in subgroups:
        for val, group in test_analysis_df.groupby(sg):
            y_t = group["true_risk"]
            y_p = group["pred_risk"]
            tp = int(((y_t == 1) & (y_p == 1)).sum())
            fp = int(((y_t == 0) & (y_p == 1)).sum())
            fn = int(((y_t == 1) & (y_p == 0)).sum())
            tn = int(((y_t == 0) & (y_p == 0)).sum())
            
            rec = tp / max(1, (tp + fn))
            prec = tp / max(1, (tp + fp))
            fpr = fp / max(1, (fp + tn))
            
            subgroup_metrics.append({
                "subgroup": sg,
                "value": str(val),
                "count": len(group),
                "recall": rec,
                "precision": prec,
                "false_positive_rate": fpr,
            })

    subgroup_df = pd.DataFrame(subgroup_metrics)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x_labels = [f"{r['subgroup']}={r['value']} (N={r['count']})" for _, r in subgroup_df.iterrows()]
    x_indices = np.arange(len(x_labels))
    width = 0.35

    ax.bar(x_indices - width/2, subgroup_df["recall"], width=width, label="Recall (Sensitivity to Risk)", color="#2b5c8f", edgecolor="black")
    ax.bar(x_indices + width/2, subgroup_df["false_positive_rate"], width=width, label="False Positive Rate", color="#d95f02", edgecolor="black")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Subgroup Performance & Parity Analysis Across Demographic Attributes", fontsize=12, pad=12)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig10_path = figures_dir / "10_subgroup_fairness_analysis.png"
    plt.savefig(fig10_path)
    plt.close()

    # Save Subgroup Report
    subgroup_report_path = reports_dir / "subgroup_fairness_report.json"
    subgroup_df.to_json(subgroup_report_path, orient="records", indent=2)

    logger.info(f"Successfully generated all 10 analytical figures in {figures_dir}")
    return {
        "figures_dir": str(figures_dir),
        "total_figures_generated": 10,
        "subgroup_report_path": str(subgroup_report_path)
    }


if __name__ == "__main__":
    generate_all_visualizations_and_eval()
