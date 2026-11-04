# Customer Churn Prediction & Subscription Analytics

A local, four-module Streamlit application for telecom churn analysis. It creates a realistic 1,200-record sample dataset in INR the first time it runs, keeps customer changes in `churn_data.csv`, and trains Logistic Regression, Random Forest, and XGBoost models.

## Run locally

```powershell
cd "E:\summer internship"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On first launch, `data_manager.py` writes `churn_data.csv` beside the Python files. The app reads and updates that local CSV; the generated file is not required to be committed.

## Run as a website

Streamlit serves the dashboard as a browser-based website. For a hosted deployment, publish this project directory to a Streamlit-compatible hosting service, install `requirements.txt`, and set `app.py` as the entry point. The included `.streamlit/config.toml` configures the wide layout's light executive theme and headless server settings. If the host uses an ephemeral filesystem, customer changes to the CSV may not survive a restart; use a host with persistent disk or connect durable storage before relying on hosted CRUD data.

To build an installable Python package from this directory:

```powershell
python -m pip install .
```

The project metadata lives in `pyproject.toml`. For the browser app, install `requirements.txt` and launch with Streamlit as shown above.

## Dashboard views

1. **Executive Dashboard** — contract and internet service slicers, customer/churn/revenue KPIs, and four interactive charts.
2. **Customer Data Explorer** — add or delete CSV records, filter the portfolio, and review predicted churn scores with risk highlighting.
3. **ML Model Evaluation** — holdout metrics, model ROC curves, and XGBoost feature importance.
4. **What-If Retention Simulator** — adjust a discount and offer duration to view the score change and estimated discount cost.

## Architecture

- `data_manager.py` — sample generation, CSV I/O, customer CRUD, and categorical/numeric preprocessing.
- `ml_engine.py` — stratified training split, train-fold SMOTE, three classifiers, metrics, probabilities, and offer scenarios.
- `visualizer.py` — Plotly Express and Graph Objects charts with a consistent executive palette.
- `app.py` — Streamlit navigation, slicers, KPI cards, forms, and dashboard layout.

The holdout split is kept separate from SMOTE. The retention simulator changes the modeled monthly charge while holding other profile fields constant; it is a directional model scenario, not a causal estimate of discount effectiveness.
