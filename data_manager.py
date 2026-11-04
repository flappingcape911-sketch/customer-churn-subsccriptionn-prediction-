"""Local CSV data generation, loading, CRUD, and feature preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "churn_data.csv"

CATEGORICAL_FEATURES = [
    "Gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Churn"


def generate_customer_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create plausible telecom subscription records with an explainable churn signal."""
    rng = np.random.default_rng(seed)
    customer_id = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    gender = rng.choice(["Female", "Male"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", rng.choice(["Yes", "No"], n, p=[0.48, 0.52]), rng.choice(["Yes", "No"], n, p=[0.18, 0.82]))
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.56, 0.24, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.48, 0.38, 0.14])
    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n, p=[0.42, 0.58]))
    def service(prob: float) -> np.ndarray:
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[prob, 1 - prob]))
    security, backup, protection, support = service(.36), service(.43), service(.40), service(.35)
    tv, movies = service(.48), service(.46)
    paperless = rng.choice(["Yes", "No"], n, p=[.62, .38])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n, p=[.34, .18, .25, .23])

    monthly = (rng.normal(78, 23, n) + (internet == "Fiber optic") * 17 - (internet == "No") * 43 + (multi == "Yes") * 8).clip(18, 180).round(2)
    total = (monthly * tenure * rng.normal(1, .035, n)).clip(0).round(2)

    # More tenure, longer contracts, and support services lower risk; billing friction and higher costs raise it.
    logit = (-1.15 + (contract == "Month-to-month") * 1.32 + (contract == "One year") * .18
             - (contract == "Two year") * .95 - tenure * .035 + (internet == "Fiber optic") * .30
             + (payment == "Electronic check") * .50 + (support == "No") * .25
             + (security == "No") * .24 + (paperless == "Yes") * .12 + senior * .20
             + (monthly - 75) * .008 + rng.normal(0, .42, n))
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "Yes", "No")
    return pd.DataFrame({
        "customerID": customer_id, "Gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
    })


def ensure_data(path: str | Path = DATA_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_customer_data().to_csv(path, index=False)
    return path


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = ensure_data(path)
    frame = pd.read_csv(path)
    return clean_data(frame)


def save_data(frame: pd.DataFrame, path: str | Path = DATA_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_data(frame).to_csv(path, index=False)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for col in FEATURES:
        if col not in data:
            data[col] = 0 if col in NUMERIC_FEATURES or col == "SeniorCitizen" else "No"
    if "customerID" not in data:
        data.insert(0, "customerID", [f"CUST-{i:05d}" for i in range(1, len(data) + 1)])
    data["customerID"] = data["customerID"].astype(str)
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    data["SeniorCitizen"] = pd.to_numeric(data["SeniorCitizen"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if TARGET not in data:
        data[TARGET] = "No"
    data[TARGET] = data[TARGET].fillna("No").astype(str).str.title().where(lambda s: s.isin(["Yes", "No"]), "No")
    return data


def add_customer(record: dict[str, Any], path: str | Path = DATA_PATH) -> pd.DataFrame:
    data = load_data(path)
    record = dict(record)
    numeric_ids = pd.to_numeric(data["customerID"].str.extract(r"(\d+)")[0], errors="coerce")
    next_id = int(numeric_ids.max() if numeric_ids.notna().any() else 0) + 1
    record.setdefault("customerID", f"CUST-{next_id:05d}")
    updated = pd.concat([data, clean_data(pd.DataFrame([record]))], ignore_index=True)
    save_data(updated, path)
    return updated


def delete_customer(customer_id: str, path: str | Path = DATA_PATH) -> pd.DataFrame:
    data = load_data(path)
    updated = data.loc[data["customerID"] != str(customer_id)].copy()
    save_data(updated, path)
    return updated


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categories and standardize numeric fields."""
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


def preprocess_data(frame: pd.DataFrame, fit: bool = True, preprocessor: ColumnTransformer | None = None):
    data = clean_data(frame)
    transformer = preprocessor or build_preprocessor()
    X = transformer.fit_transform(data[FEATURES]) if fit else transformer.transform(data[FEATURES])
    return X, data[TARGET].map({"No": 0, "Yes": 1}).to_numpy(), transformer

