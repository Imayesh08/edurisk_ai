"""
Model Training and Experimentation Pipeline for EduRisk AI.
Trains reference baselines (DummyClassifier, Logistic Regression),
ensemble models (Random Forest), and gradient boosters (XGBoost, LightGBM).
Performs PCA dimensionality reduction analysis, hyperparameter tuning with Optuna,
and saves validated pipeline artifacts.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.features.preprocessing import prepare_and_split_data, create_preprocessor

# Configure Optuna logging level
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("EduRisk-Train")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }


def train_and_evaluate_all():
    """Executes full training pipeline, PCA experiments, Optuna tuning, and evaluation."""
    config = load_config()
    artifacts_dir = PROJECT_ROOT / config["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = prepare_and_split_data()

    # 2. Fit Preprocessor ONLY on training data (Zero Data Leakage)
    preprocessor = create_preprocessor(metadata["num_features"], metadata["cat_features"])
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    logger.info(f"Transformed feature count: {len(feature_names)}")

    # Imbalance ratio for gradient boosting scale_pos_weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / max(1, pos_count))
    logger.info(f"Class counts in training: 0={neg_count}, 1={pos_count}. scale_pos_weight={scale_pos_weight:.2f}")

    # 3. Model Benchmark Dictionary
    results = {}
    models_dict = {}

    # --- Model 1: DummyClassifier (Baseline) ---
    logger.info("Training 1. DummyClassifier Baseline...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train_trans, y_train)
    dummy_proba = dummy.predict_proba(X_test_trans)[:, 1]
    dummy_pred = dummy.predict(X_test_trans)
    results["DummyClassifier"] = compute_metrics(y_test, dummy_pred, dummy_proba)
    models_dict["DummyClassifier"] = dummy

    # --- Model 2: Logistic Regression (Interpretable Baseline) ---
    logger.info("Training 2. Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.5)
    lr.fit(X_train_trans, y_train)
    lr_proba = lr.predict_proba(X_test_trans)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    results["LogisticRegression"] = compute_metrics(y_test, lr_pred, lr_proba)
    models_dict["LogisticRegression"] = lr

    # --- Model 3: Random Forest Classifier ---
    logger.info("Training 3. Random Forest (balanced_subsample)...")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_split=4, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42
    )
    rf.fit(X_train_trans, y_train)
    rf_proba = rf.predict_proba(X_test_trans)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    results["RandomForest"] = compute_metrics(y_test, rf_pred, rf_proba)
    models_dict["RandomForest"] = rf

    # --- Model 4: LightGBM Classifier ---
    logger.info("Training 4. LightGBM...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=15,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
    )
    lgbm.fit(X_train_trans, y_train)
    lgbm_proba = lgbm.predict_proba(X_test_trans)[:, 1]
    lgbm_pred = (lgbm_proba >= 0.5).astype(int)
    results["LightGBM"] = compute_metrics(y_test, lgbm_pred, lgbm_proba)
    models_dict["LightGBM"] = lgbm

    # --- Model 5: XGBoost (Default balanced) ---
    logger.info("Training 5. XGBoost (Standard)...")
    xgb_default = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
    )
    xgb_default.fit(X_train_trans, y_train)
    xgb_default_proba = xgb_default.predict_proba(X_test_trans)[:, 1]
    xgb_default_pred = (xgb_default_proba >= 0.5).astype(int)
    results["XGBoost_Default"] = compute_metrics(y_test, xgb_default_pred, xgb_default_proba)

    # =========================================================================
    # EXPERIMENT: PCA Dimensionality Reduction vs Original Features
    # =========================================================================
    logger.info("Running PCA Dimensionality Reduction Experiment (95% variance)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_trans)
    X_test_pca = pca.transform(X_test_trans)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    n_pca_components = int(pca.n_components_)
    logger.info(f"PCA reduced {X_train_trans.shape[1]} features to {n_pca_components} components (Variance: {explained_variance:.1%})")

    # Logistic Regression with PCA
    lr_pca = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.5)
    lr_pca.fit(X_train_pca, y_train)
    lr_pca_proba = lr_pca.predict_proba(X_test_pca)[:, 1]
    lr_pca_pred = (lr_pca_proba >= 0.5).astype(int)
    results["LogisticRegression_PCA"] = compute_metrics(y_test, lr_pca_pred, lr_pca_proba)

    # Random Forest with PCA
    rf_pca = RandomForestClassifier(
        n_estimators=150, max_depth=6, class_weight="balanced_subsample", random_state=42
    )
    rf_pca.fit(X_train_pca, y_train)
    rf_pca_proba = rf_pca.predict_proba(X_test_pca)[:, 1]
    rf_pca_pred = (rf_pca_proba >= 0.5).astype(int)
    results["RandomForest_PCA"] = compute_metrics(y_test, rf_pca_pred, rf_pca_proba)

    pca_experiment_data = {
        "n_original_features": int(X_train_trans.shape[1]),
        "n_pca_components": n_pca_components,
        "total_explained_variance": explained_variance,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance_ratio": [float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
    }

    # =========================================================================
    # HYPERPARAMETER OPTIMIZATION (Optuna on XGBoost)
    # =========================================================================
    logger.info("Running Optuna Hyperparameter Optimization on XGBoost...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_pos_weight * 0.8, scale_pos_weight * 1.5),
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_trans, y_train)
        val_proba = model.predict_proba(X_val_trans)[:, 1]
        # Optimize PR-AUC (Average Precision) on validation set
        return average_precision_score(y_val, val_proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=config["models"]["optuna_trials"])
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["eval_metric"] = "logloss"
    logger.info(f"Optuna Best Validation PR-AUC: {study.best_value:.4f}")
    logger.info(f"Optuna Best Parameters: {best_params}")

    # Train final tuned XGBoost
    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_train_trans, y_train)
    best_xgb_proba = best_xgb.predict_proba(X_test_trans)[:, 1]

    # Threshold Optimization Analysis (Decision Threshold for Risk Sensitivity)
    thresholds = np.linspace(0.1, 0.9, 17)
    threshold_analysis = []
    for th in thresholds:
        th_pred = (best_xgb_proba >= th).astype(int)
        rec = recall_score(y_test, th_pred, zero_division=0)
        prec = precision_score(y_test, th_pred, zero_division=0)
        f1 = f1_score(y_test, th_pred, zero_division=0)
        threshold_analysis.append({
            "threshold": float(th),
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
        })

    # Select operational threshold (Configured or F1-optimal)
    best_th = config["models"]["decision_threshold"]
    best_xgb_pred = (best_xgb_proba >= best_th).astype(int)
    results["XGBoost_Tuned"] = compute_metrics(y_test, best_xgb_pred, best_xgb_proba)
    models_dict["XGBoost_Tuned"] = best_xgb

    # Print Comparative Results Table
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 95)
    print("                      MODEL BENCHMARK RESULTS ON HELD-OUT TEST SET (N=130)")
    print("=" * 95)
    print(results_df[["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]].to_string())
    print("=" * 95 + "\n")

    # =========================================================================
    # SAVE PRODUCTION ARTIFACTS
    # =========================================================================
    # Best Model is Tuned XGBoost
    best_model_path = PROJECT_ROOT / config["paths"]["best_model_path"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor_path"]

    joblib.dump(best_xgb, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(pca, artifacts_dir / "pca_transformer.joblib")
    joblib.dump(rf, artifacts_dir / "random_forest_model.joblib")
    joblib.dump(lr, artifacts_dir / "logistic_regression_model.joblib")

    # Save complete evaluation bundle
    eval_bundle = {
        "metadata": metadata,
        "feature_names": feature_names,
        "benchmark_results": results,
        "pca_experiment": pca_experiment_data,
        "optuna_best_params": best_params,
        "threshold_analysis": threshold_analysis,
        "selected_threshold": best_th,
        "confusion_matrix": confusion_matrix(y_test, best_xgb_pred).tolist(),
        "test_probabilities": best_xgb_proba.tolist(),
        "test_ground_truth": y_test.tolist(),
        "test_predictions": best_xgb_pred.tolist(),
        "model_version": config["project"]["version"],
    }

    metrics_path = PROJECT_ROOT / config["paths"]["metrics_path"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_bundle, f, indent=2)

    logger.info(f"Saved best model artifact to: {best_model_path}")
    logger.info(f"Saved preprocessor artifact to: {preprocessor_path}")
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    return eval_bundle


if __name__ == "__main__":
    train_and_evaluate_all()
