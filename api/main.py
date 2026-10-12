"""
Production FastAPI Inference Application for EduRisk AI.
Provides RESTful APIs for individual and batch risk prediction,
model provenance metadata, evaluation metrics, and health monitoring.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.models.predict import EduRiskPredictor
from api.schemas import (
    StudentProfileInput,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    HealthResponse,
    ModelInfoResponse,
)
from api.dependencies import get_predictor

config = load_config()

app = FastAPI(
    title=f"{config['project']['name']} Inference Service",
    version=config["project"]["version"],
    description=(
        "Research-grade API for explainable, early-stage prediction of student academic risk. "
        "Built for 19MAM56 Advanced Machine Learning Lab CAT II."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend/dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System Health"])
def health_check(predictor: EduRiskPredictor = Depends(get_predictor)):
    """Health check endpoint to verify service readiness and artifact loading."""
    return HealthResponse(
        status="healthy",
        version=config["project"]["version"],
        model_loaded=predictor.model is not None,
        pipeline_ready=predictor.preprocessor is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Provenance"])
def get_model_info():
    """Returns model architecture, version, features, and validated test performance."""
    metrics_path = PROJECT_ROOT / config["paths"]["metrics_path"]
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metrics artifact not found. Please train models first.",
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    tuned_metrics = metrics["benchmark_results"].get("XGBoost_Tuned", {})

    return ModelInfoResponse(
        model_name="Tuned XGBoost (Early Warning Classifier)",
        version=config["project"]["version"],
        decision_threshold=config["models"]["decision_threshold"],
        primary_metric="PR-AUC (Average Precision) & Minority Recall",
        test_roc_auc=tuned_metrics.get("roc_auc", 0.0),
        test_pr_auc=tuned_metrics.get("pr_auc", 0.0),
        test_recall_minority=tuned_metrics.get("recall", 0.0),
        numerical_features=metrics["metadata"]["num_features"],
        categorical_features=metrics["metadata"]["cat_features"],
    )


@app.get("/metrics", tags=["Evaluation & Benchmarks"])
def get_benchmark_metrics():
    """Returns complete comparative benchmark results and PCA experiment logs."""
    metrics_path = PROJECT_ROOT / config["paths"]["metrics_path"]
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics file unavailable.",
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single_student(
    student: StudentProfileInput,
    threshold: Optional[float] = Query(None, ge=0.05, le=0.95, description="Custom decision threshold"),
    predictor: EduRiskPredictor = Depends(get_predictor),
):
    """
    Performs inference on an individual student profile with SHAP feature explanations.
    """
    try:
        student_data = student.model_dump()
        result = predictor.predict_student(student_data, threshold=threshold)
        return SinglePredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_student_batch(
    request: BatchPredictionRequest,
    predictor: EduRiskPredictor = Depends(get_predictor),
):
    """
    Performs batch inference on a cohort of student profiles.
    """
    try:
        raw_rows = [s.model_dump() for s in request.students]
        df_batch = pd.DataFrame(raw_rows)
        results_df = predictor.predict_batch(df_batch, threshold=request.threshold)

        items = []
        for idx, row in results_df.iterrows():
            items.append(
                BatchPredictionItem(
                    student_index=int(idx),
                    predicted_risk_probability=float(row["predicted_risk_probability"]),
                    predicted_academic_risk=str(row["predicted_academic_risk"]),
                    risk_tier=str(row["risk_tier"]),
                )
            )

        total_processed = len(items)
        at_risk_count = sum(1 for it in items if it.predicted_academic_risk == "At Risk")
        at_risk_rate = at_risk_count / max(1, total_processed)

        return BatchPredictionResponse(
            total_processed=total_processed,
            total_at_risk=at_risk_count,
            at_risk_percentage=round(at_risk_rate, 4),
            predictions=items,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config["api"]["host"], port=config["api"]["port"], reload=True)
