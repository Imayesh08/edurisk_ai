"""
Tests for FastAPI inference endpoints using Starlette/FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_STUDENT = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 3,
    "Fedu": 2,
    "Mjob": "services",
    "Fjob": "other",
    "reason": "reputation",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2,
    "G1": 12
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["pipeline_ready"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "XGBoost" in data["model_name"]
    assert data["test_roc_auc"] > 0.8
    assert data["test_recall_minority"] >= 0.7


def test_predict_single_endpoint():
    response = client.post("/predict", json=SAMPLE_STUDENT)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_status"] in ["At Risk", "Lower Risk"]
    assert len(data["top_risk_drivers"]) > 0 or len(data["top_protective_factors"]) > 0
    assert len(data["evidence_based_interventions"]) > 0


def test_predict_batch_endpoint():
    batch_payload = {
        "students": [SAMPLE_STUDENT, {**SAMPLE_STUDENT, "G1": 6, "failures": 2, "absences": 15}]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    # Second student with failures and G1=6 should have high risk
    assert data["predictions"][1]["predicted_risk_probability"] > data["predictions"][0]["predicted_risk_probability"]
