"""
Inference and Prediction Service for EduRisk AI.
Provides single-student and batch predictions with probability scores,
calibrated risk decisions, and integrated local SHAP explainability.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.features.preprocessing import EducationalFeatureEngineer
from src.explainability.shap_explainer import EduRiskExplainer


class EduRiskPredictor:
    """
    Production inference engine for predicting student academic risk.
    """

    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        self.config = load_config()
        m_path = Path(model_path) if model_path else PROJECT_ROOT / self.config["paths"]["best_model_path"]
        p_path = Path(preprocessor_path) if preprocessor_path else PROJECT_ROOT / self.config["paths"]["preprocessor_path"]

        if not m_path.exists() or not p_path.exists():
            raise FileNotFoundError("Model or preprocessor pipeline not found. Run training first.")

        self.model = joblib.load(m_path)
        self.preprocessor = joblib.load(p_path)
        self.feature_engineer = EducationalFeatureEngineer()
        self.default_threshold = float(self.config["models"]["decision_threshold"])
        self.explainer = EduRiskExplainer(str(m_path), str(p_path))

    def predict_student(self, student_dict: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
        """
        Runs inference on a single student profile.
        Returns risk probability, risk category, key risk drivers, and interventions.
        """
        th = threshold if threshold is not None else self.default_threshold
        df = pd.DataFrame([student_dict])
        
        # Apply domain feature engineering
        df_engineered = self.feature_engineer.transform(df)
        X_trans = self.preprocessor.transform(df_engineered)

        proba = float(self.model.predict_proba(X_trans)[0, 1])
        is_at_risk = bool(proba >= th)
        risk_level = "High Risk" if proba >= 0.65 else ("Moderate Risk" if proba >= th else "Low Risk")

        # Generate local explanation
        explanation = self.explainer.explain_local_instance(df_engineered)

        return {
            "prediction": int(is_at_risk),
            "risk_status": "At Risk" if is_at_risk else "Lower Risk",
            "risk_level": risk_level,
            "risk_probability": round(proba, 4),
            "applied_threshold": th,
            "confidence_assessment": f"Estimated risk probability of {proba:.1%} exceeds critical alert threshold ({th:.1%})" if is_at_risk else f"Estimated risk probability of {proba:.1%} is within safe limits (threshold {th:.1%})",
            "top_risk_drivers": explanation["top_risk_drivers"],
            "top_protective_factors": explanation["top_protective_factors"],
            "evidence_based_interventions": explanation["evidence_based_interventions"],
            "epistemic_disclaimer": explanation["epistemic_disclaimer"],
        }

    def predict_batch(self, df_raw: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Runs batch inference across a dataframe of student records.
        """
        th = threshold if threshold is not None else self.default_threshold
        df_engineered = self.feature_engineer.transform(df_raw.copy())
        X_trans = self.preprocessor.transform(df_engineered)

        probas = self.model.predict_proba(X_trans)[:, 1]
        preds = (probas >= th).astype(int)

        out_df = df_raw.copy()
        out_df["predicted_risk_probability"] = np.round(probas, 4)
        out_df["predicted_academic_risk"] = ["At Risk" if p == 1 else "Lower Risk" for p in preds]
        out_df["risk_tier"] = [
            "High Risk" if pr >= 0.65 else ("Moderate Risk" if pr >= th else "Low Risk")
            for pr in probas
        ]
        return out_df
