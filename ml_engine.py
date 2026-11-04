"""Model training, evaluation, probability scoring, and retention what-if analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from data_manager import FEATURES, TARGET, build_preprocessor, clean_data


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=220, max_depth=4, learning_rate=.045,
                             subsample=.86, colsample_bytree=.86, reg_lambda=1.2,
                             eval_metric="logloss", random_state=42, n_jobs=-1)
    except ImportError as exc:
        raise RuntimeError("XGBoost is required. Install project dependencies with `pip install -r requirements.txt`.") from exc


def train_models(frame: pd.DataFrame) -> dict:
    """Fit all requested classifiers after train/test splitting and train-only SMOTE."""
    data = clean_data(frame)
    X_raw, y = data[FEATURES], data[TARGET].map({"No": 0, "Yes": 1}).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Model training needs at least one churned and one retained customer.")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=.25, random_state=42, stratify=y)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    # SMOTE is applied only to the training fold to avoid test-set leakage.
    min_class = int(np.bincount(y_train).min())
    if min_class >= 2:
        X_fit, y_fit = SMOTE(random_state=42, k_neighbors=min(5, min_class - 1)).fit_resample(X_train, y_train)
    else:
        warnings.warn("SMOTE skipped because the training split has fewer than two minority samples.")
        X_fit, y_fit = X_train, y_train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight=None, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 class_weight=None, random_state=42, n_jobs=-1),
        "XGBoost": _xgb_classifier(),
    }
    metrics, curves = {}, {}
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= .5).astype(int)
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-Score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        curves[name] = {"fpr": fpr, "tpr": tpr, "auc": metrics[name]["ROC-AUC"]}
    best_model = models["XGBoost"]
    importance = pd.DataFrame({"Feature": feature_names, "Importance": best_model.feature_importances_}).sort_values("Importance", ascending=False)
    return {"models": models, "preprocessor": preprocessor, "metrics": metrics,
            "roc_curves": curves, "feature_importance": importance,
            "feature_names": feature_names, "best_model": best_model}


def predict_churn(frame: pd.DataFrame, model_bundle: dict, model_name: str = "XGBoost") -> np.ndarray:
    data = clean_data(frame)
    X = model_bundle["preprocessor"].transform(data[FEATURES])
    return model_bundle["models"][model_name].predict_proba(X)[:, 1]


def simulate_retention_offer(customer: pd.DataFrame | pd.Series | dict, discount_percent: float, model_bundle: dict) -> dict:
    """Score a customer's current plan and its discounted monthly-price scenario."""
    if isinstance(customer, pd.Series):
        customer_frame = customer.to_frame().T
    elif isinstance(customer, dict):
        customer_frame = pd.DataFrame([customer])
    else:
        customer_frame = customer.iloc[[0]].copy()
    base = clean_data(customer_frame)
    offered = base.copy()
    discount_percent = float(np.clip(discount_percent, 0, 50))
    offered.loc[:, "MonthlyCharges"] = offered["MonthlyCharges"] * (1 - discount_percent / 100)
    # Preserve historic billed total; the simulation changes only the future monthly rate.
    current = float(predict_churn(base, model_bundle)[0])
    after = float(predict_churn(offered, model_bundle)[0])
    monthly = float(base["MonthlyCharges"].iloc[0])
    return {"current_probability": current, "offer_probability": after,
            "risk_reduction_pp": (current - after) * 100,
            "monthly_savings": monthly * discount_percent / 100,
            "new_monthly_charge": monthly * (1 - discount_percent / 100)}

