"""
Explainable AI (XAI) Module for EduRisk AI.
Integrates SHAP (TreeExplainer) to generate mathematically rigorous,
global and local explanations of academic risk predictions.
Distinguishes statistical association from causal attribution.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config

logger = logging.getLogger("EduRisk-XAI")


class EduRiskExplainer:
    """
    SHAP-based explainer for academic risk models.
    Provides global feature attribution and individual student local explanations.
    """

    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        config = load_config()
        m_path = Path(model_path) if model_path else PROJECT_ROOT / config["paths"]["best_model_path"]
        p_path = Path(preprocessor_path) if preprocessor_path else PROJECT_ROOT / config["paths"]["preprocessor_path"]

        if not m_path.exists() or not p_path.exists():
            raise FileNotFoundError(f"Model or Preprocessor artifact missing at {m_path} or {p_path}")

        self.model = joblib.load(m_path)
        self.preprocessor = joblib.load(p_path)
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def explain_global(self, X_transformed: np.ndarray, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates global SHAP values across dataset and saves summary figures.
        """
        config = load_config()
        out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / config["paths"]["figures_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        shap_values = self.explainer(X_transformed)

        # Handle binary classification SHAP shapes
        vals = shap_values.values
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        # Save Global SHAP Summary Plot
        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            vals,
            X_transformed,
            feature_names=self.feature_names,
            max_display=15,
            show=False
        )
        plt.title("SHAP Global Feature Importance (Academic Risk Predictors)", fontsize=13, pad=15)
        plt.tight_layout()
        summary_plot_path = out_dir / "shap_summary_plot.png"
        plt.savefig(summary_plot_path, bbox_inches="tight")
        plt.close()

        # Save Global Feature Importance Bar Plot
        plt.figure(figsize=(10, 6), dpi=300)
        top15 = importance_df.head(15).iloc[::-1]
        plt.barh(top15["feature"], top15["mean_abs_shap"], color="#1e88e5", edgecolor="#0d47a1")
        plt.xlabel("Mean |SHAP Value| (Impact on Risk Model Output)", fontsize=11)
        plt.title("Top 15 Predictors of Student Academic Risk", fontsize=13)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        bar_plot_path = out_dir / "shap_importance_bar.png"
        plt.savefig(bar_plot_path, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved global SHAP plots to {out_dir}")

        return {
            "top_features": importance_df.head(15).to_dict(orient="records"),
            "summary_plot_path": str(summary_plot_path),
            "bar_plot_path": str(bar_plot_path),
        }

    def explain_local_instance(self, single_raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates local explanation for a single student profile.
        Returns top risk-increasing and risk-decreasing factors with human-readable guidance.
        """
        # Transform through trained pipeline
        X_trans = self.preprocessor.transform(single_raw_df)
        pred_proba = float(self.model.predict_proba(X_trans)[0, 1])

        shap_values = self.explainer(X_trans)
        vals = shap_values.values[0]
        if len(vals.shape) == 2:
            vals = vals[:, 1]

        # Base value (expected model output)
        base_val = float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value[1])

        feature_contributions = []
        for name, shap_val, trans_val in zip(self.feature_names, vals, X_trans[0]):
            feature_contributions.append({
                "feature": name,
                "shap_value": float(shap_val),
                "transformed_value": float(trans_val),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                "abs_impact": float(abs(shap_val)),
            })

        # Sort by absolute impact
        feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_risk_drivers = [f for f in feature_contributions if f["shap_value"] > 0][:5]
        top_protective_factors = [f for f in feature_contributions if f["shap_value"] < 0][:5]

        # Generate evidence-based academic intervention suggestions
        interventions = self._generate_interventions(single_raw_df.iloc[0], top_risk_drivers)

        return {
            "predicted_risk_probability": pred_proba,
            "base_value": base_val,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "evidence_based_interventions": interventions,
            "epistemic_disclaimer": (
                "These explanations reflect statistical associations learned by the machine learning model "
                "from historical cohort data. They do NOT establish direct causality. This assessment is an "
                "academic decision-support tool to assist educators in allocating proactive support, not an automated "
                "judgment or deterministic outcome."
            ),
        }

    def _generate_interventions(self, student_row: pd.Series, top_drivers: List[Dict[str, Any]]) -> List[str]:
        """Generates evidence-backed educational support actions."""
        recommendations = []
        driver_names = [d["feature"].lower() for d in top_drivers]

        if any("failures" in d or "academic_friction" in d for d in driver_names) and student_row.get("failures", 0) > 0:
            recommendations.append(
                f"Curricular Remediation: Student has recorded {student_row.get('failures')} prior academic failure(s). "
                "Recommend diagnostic skills assessment and enrollment in peer-assisted foundational tutoring."
            )

        if any("g1" in d for d in driver_names) and student_row.get("G1", 20) < 10:
            recommendations.append(
                f"Early Academic Intervention: First period grade is {student_row.get('G1')}/20. "
                "Schedule a 1-on-1 feedback session before midterm assessments to address core concept gaps."
            )

        if any("absences" in d for d in driver_names) and student_row.get("absences", 0) > 4:
            recommendations.append(
                f"Attendance Outreach: Student has accumulated {student_row.get('absences')} absences. "
                "Initiate academic advisor check-in to identify potential transportation, health, or scheduling barriers."
            )

        if any("study" in d for d in driver_names) and student_row.get("studytime", 4) <= 2:
            recommendations.append(
                "Study Skills Coaching: Weekly self-study time is under 5 hours. "
                "Provide time-management workshop resources and structured campus study group pairing."
            )

        if any("alcohol" in d or "social" in d or "goout" in d for d in driver_names):
            recommendations.append(
                "Student Wellness & Holistic Support: High social distraction / lifestyle indicators detected. "
                "Facilitate voluntary student affairs counseling and campus engagement mentorship."
            )

        if not recommendations:
            recommendations.append(
                "General Advisory Monitoring: No high-criticality risk flags detected. Continue routine semester progress reviews."
            )

        return recommendations
