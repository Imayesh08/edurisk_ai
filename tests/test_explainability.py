"""
Tests for SHAP explainer module.
"""

import pytest
import pandas as pd
from src.explainability.shap_explainer import EduRiskExplainer
from src.features.preprocessing import EducationalFeatureEngineer
from tests.test_api import SAMPLE_STUDENT


def test_explainer_initialization_and_local_explanation():
    explainer = EduRiskExplainer()
    engineer = EducationalFeatureEngineer()
    
    df = pd.DataFrame([SAMPLE_STUDENT])
    df_eng = engineer.transform(df)
    
    explanation = explainer.explain_local_instance(df_eng)
    assert "predicted_risk_probability" in explanation
    assert "top_risk_drivers" in explanation
    assert "top_protective_factors" in explanation
    assert "evidence_based_interventions" in explanation
    assert "epistemic_disclaimer" in explanation
    assert len(explanation["evidence_based_interventions"]) > 0
