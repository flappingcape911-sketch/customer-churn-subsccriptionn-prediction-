"""Executive customer churn and subscription analytics dashboard."""

from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_manager import DATA_PATH, add_customer, delete_customer, load_data
from ml_engine import predict_churn, simulate_retention_offer, train_models
from visualizer import (BLUE, GOLD, NAVY, RED, TEAL, charges_histogram,
                        churn_donut, contract_churn_bar, feature_importance_chart,
                        roc_auc_chart, tenure_risk_trend)


st.set_page_config(page_title="Customer Intelligence | Churn & Subscription Analytics",
                   page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#17365D; --blue:#2F75B5; --teal:#00A6A6; --gold:#F2A541; --ink:#24364B; }
.stApp { background:#F3F6FA; color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#132D4E; border-right:1px solid #294464; }
[data-testid="stSidebar"] * { color:#EDF4FA; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:8px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--navy); }
.hero { background:linear-gradient(112deg,#17365D 0%,#245787 70%,#168B9A 130%); padding:25px 30px; border-radius:16px; color:#fff; margin:2px 0 18px; box-shadow:0 10px 26px rgba(23,54,93,.16); }
.hero h1 { color:white !important; margin:0; font-size:28px; letter-spacing:-.5px; }
.hero p { color:#D9E9F3; margin:5px 0 0; font-size:13px; }
.section-label { font-size:11px; letter-spacing:1.3px; font-weight:700; color:#71839A; text-transform:uppercase; margin:12px 0 8px; }
.kpi-card { background:#fff; border:1px solid #E2E8EF; border-top:4px solid var(--accent); border-radius:12px; padding:16px 18px 14px; box-shadow:0 4px 14px rgba(23,54,93,.055); min-height:116px; }
.kpi-label { font-size:12px; color:#63758A; font-weight:600; }
.kpi-value { font:800 26px 'Manrope',sans-serif; color:#1E3550; margin:8px 0 3px; }
.kpi-note { font-size:11px; color:#8A98A8; }
.card-head { background:#fff; border:1px solid #E4EAF0; border-bottom:0; border-radius:12px 12px 0 0; padding:15px 18px 0; }
.stPlotlyChart { background:#fff; border:1px solid #E4EAF0; border-top:0; border-radius:0 0 12px 12px; padding:0 8px 4px; box-shadow:0 4px 14px rgba(23,54,93,.045); }
div[data-testid="stForm"], div[data-testid="stDataFrame"] { border-radius:10px; }
.footnote { color:#8290A0; font-size:11px; padding-top:10px; }
div[data-testid="stMetric"] { background:white; border:1px solid #E3E9F0; border-left:4px solid #00A6A6; padding:14px 16px; border-radius:10px; box-shadow:0 4px 12px rgba(23,54,93,.05); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training churn models…")
def cached_models(data: pd.DataFrame, fingerprint: str):
    return train_models(data)


def kpi_card(label: str, value: str, note: str, accent: str):
    st.markdown(f'<div class="kpi-card" style="--accent:{accent}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def header(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def chart(title: str, fig, key: str):
    st.markdown(f'<div class="card-head"><div style="font-size:14px;font-weight:700;color:{NAVY}">{title}</div></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=key)


data = load_data()
fp = hashlib.sha1(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()
try:
    bundle = cached_models(data, fp)
except Exception as exc:
    st.error(f"Model setup failed: {exc}")
    st.info("Confirm the packages in requirements.txt are installed, then restart the app.")
    st.stop()

with st.sidebar:
    st.markdown("<div style='padding:8px 4px 18px'><div style='font-size:11px;letter-spacing:1.5px;color:#8EB9D6;font-weight:700'>CUSTOMER INTELLIGENCE</div><div style='font:800 20px Manrope;color:#fff;margin-top:6px'>SUBSCRIPTION<br>ANALYTICS</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#8EB9D6;letter-spacing:1.4px;margin:6px 0'>WORKSPACE</div>", unsafe_allow_html=True)
    view = st.radio("Navigate", ["Executive Dashboard", "Customer Data Explorer", "ML Model Evaluation", "What-If Retention Simulator"], label_visibility="collapsed")
    st.markdown("<div style='border-top:1px solid #34506F;margin:22px 0 12px'></div>", unsafe_allow_html=True)
    st.caption(f"LOCAL DATA · {len(data):,} CUSTOMERS")
    st.caption(f"Source: {DATA_PATH.name}")


if view == "Executive Dashboard":
    header("Customer Churn & Subscription Analytics", "Executive overview  ·  Telecom portfolio  ·  INR (₹)")
    st.markdown('<div class="section-label">Portfolio filters</div>', unsafe_allow_html=True)
    s1, s2, spacer = st.columns([1, 1, 3])
    with s1:
        contracts = ["All contracts"] + sorted(data["Contract"].dropna().unique().tolist())
        contract_filter = st.selectbox("Contract type", contracts, label_visibility="collapsed")
    with s2:
        services = ["All internet services"] + sorted(data["InternetService"].dropna().unique().tolist())
        service_filter = st.selectbox("Internet service", services, label_visibility="collapsed")
    filtered = data.copy()
    if contract_filter != "All contracts":
        filtered = filtered[filtered["Contract"] == contract_filter]
    if service_filter != "All internet services":
        filtered = filtered[filtered["InternetService"] == service_filter]
    if filtered.empty:
        st.warning("No customers match these slicers.")
        st.stop()
    risks = predict_churn(filtered, bundle)
    churn_rate = filtered["Churn"].eq("Yes").mean()
    arpu = filtered["MonthlyCharges"].mean()
    revenue_at_risk = filtered.loc[filtered["Churn"].eq("Yes"), "MonthlyCharges"].sum()
    st.markdown('<div class="section-label">Executive KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("CUSTOMERS IN SCOPE", f"{len(filtered):,}", "Active customer records", BLUE)
    with k2: kpi_card("OBSERVED CHURN RATE", f"{churn_rate:.1%}", f"{int(filtered['Churn'].eq('Yes').sum()):,} churned customers", RED)
    with k3: kpi_card("AVERAGE MONTHLY CHARGE", f"₹{arpu:,.0f}", "Average revenue per customer", TEAL)
    with k4: kpi_card("MONTHLY REVENUE AT RISK", f"₹{revenue_at_risk:,.0f}", "Observed churned customers", GOLD)
    st.markdown('<div class="section-label">Portfolio performance</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left: chart("CUSTOMER MIX", churn_donut(filtered), "exec_donut")
    with right: chart("CONTRACT RETENTION", contract_churn_bar(filtered), "exec_contract")
    left, right = st.columns(2, gap="medium")
    with left: chart("RISK BY TENURE", tenure_risk_trend(filtered), "exec_tenure")
    with right: chart("CHARGES DISTRIBUTION", charges_histogram(filtered), "exec_charges")
    st.markdown('<div class="footnote">Observed churn is historical. Model scores are estimates based on the current customer profile. Currency values are in Indian rupees.</div>', unsafe_allow_html=True)

elif view == "Customer Data Explorer":
    header("Customer Data Explorer", "Maintain local customer records and inspect model-scored churn risk.")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Add a customer")
        with st.form("add_customer_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                tenure = st.number_input("Tenure (months)", 0, 72, 12)
                internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                monthly = st.number_input("Monthly charges (₹)", 0.0, 1000.0, 79.0, step=1.0)
            with c2:
                senior = st.selectbox("Senior citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                support = st.selectbox("Tech support", ["No", "Yes"] if internet != "No" else ["No internet service"])
                payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                churn = st.selectbox("Observed churn", ["No", "Yes"])
            submitted = st.form_submit_button("＋  Add customer", type="primary", use_container_width=True)
        if submitted:
            yes_no_services = lambda val: val if internet != "No" else "No internet service"
            record = {"Gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
                      "Dependents": "No", "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
                      "InternetService": internet, "OnlineSecurity": yes_no_services("No"),
                      "OnlineBackup": yes_no_services("No"), "DeviceProtection": yes_no_services("No"),
                      "TechSupport": yes_no_services(support), "StreamingTV": yes_no_services("No"),
                      "StreamingMovies": yes_no_services("No"), "Contract": contract,
                      "PaperlessBilling": "Yes", "PaymentMethod": payment,
                      "MonthlyCharges": monthly, "TotalCharges": monthly * tenure, "Churn": churn}
            add_customer(record)
            st.cache_resource.clear()
            st.success("Customer saved to churn_data.csv.")
            st.rerun()
        st.markdown("---")
        st.subheader("Delete a customer")
        selected_id = st.selectbox("Customer ID", data["customerID"].tolist(), key="delete_customer_id")
        if st.button("Delete selected record", type="secondary", use_container_width=True):
            delete_customer(selected_id)
            st.cache_resource.clear()
            st.success(f"{selected_id} deleted.")
            st.rerun()
    with right:
        st.subheader("Customer portfolio")
        filt1, filt2, filt3 = st.columns(3)
        with filt1: contract_choice = st.selectbox("Contract", ["All"] + sorted(data["Contract"].unique()), key="explore_contract")
        with filt2: churn_choice = st.selectbox("Churn", ["All", "Yes", "No"], key="explore_churn")
        with filt3: risk_floor = st.slider("Minimum risk (%)", 0, 100, 0)
        table = data.copy()
        if contract_choice != "All": table = table[table["Contract"] == contract_choice]
        if churn_choice != "All": table = table[table["Churn"] == churn_choice]
        table["Churn risk (%)"] = (predict_churn(table, bundle) * 100).round(1) if len(table) else pd.Series(dtype=float)
        table = table[table["Churn risk (%)"] >= risk_floor]
        display_cols = ["customerID", "Contract", "InternetService", "tenure", "MonthlyCharges", "Churn", "Churn risk (%)"]
        styled = table[display_cols].style.format({"MonthlyCharges": "₹{:,.2f}", "Churn risk (%)": "{:.1f}%"}).map(
            lambda value: "background-color: #FDE8E7; color:#A3312D; font-weight:700" if isinstance(value, (int, float)) and value >= 60 else "background-color:#FFF4DD; color:#875900" if isinstance(value, (int, float)) and value >= 35 else "", subset=["Churn risk (%)"])
        st.dataframe(styled, use_container_width=True, height=570, hide_index=True)
        st.caption(f"Showing {len(table):,} of {len(data):,} records · risk highlighting: amber ≥35%, red ≥60%.")

elif view == "ML Model Evaluation":
    header("Model Evaluation & Explainability", "Holdout performance, ranking quality, and leading churn drivers.")
    metrics = pd.DataFrame(bundle["metrics"]).T.reset_index(names="Model")
    st.subheader("Holdout model comparison")
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "Model"})
                 .set_properties(**{"font-weight": "600"}, subset=["Model"]),
                 use_container_width=True, hide_index=True)
    st.caption("Evaluation uses a stratified 75/25 split. SMOTE is applied to the training data only. Classification threshold: 50%.")
    left, right = st.columns([1.15, .85], gap="large")
    with left: chart("MODEL DISCRIMINATION", roc_auc_chart(bundle), "eval_roc")
    with right: chart("TOP CHURN DRIVERS", feature_importance_chart(bundle), "eval_importance")

else:
    header("What-If Retention Simulator", "Estimate how a monthly discount changes predicted churn risk for a selected customer.")
    customer_id = st.selectbox("Select customer", data["customerID"].tolist())
    customer = data.loc[data["customerID"] == customer_id].iloc[0]
    st.markdown('<div class="section-label">Customer snapshot</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, label, val in zip(cols, ["CONTRACT", "TENURE", "INTERNET", "MONTHLY CHARGE", "OBSERVED CHURN"],
                               [customer["Contract"], f"{int(customer['tenure'])} months", customer["InternetService"], f"₹{customer['MonthlyCharges']:,.2f}", customer["Churn"]]):
        with col: kpi_card(label, str(val), f"Customer {customer_id}" if label == "CONTRACT" else "Current profile", BLUE if label != "OBSERVED CHURN" else RED)
    st.markdown('<div class="section-label">Retention offer</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([.8, 1.2], gap="large")
    with c1:
        discount = st.slider("Monthly discount offer", 0, 50, 10, step=1, format="%d%%")
        duration = st.slider("Offer duration", 1, 12, 6, help="Used to estimate customer savings; model score reflects the new monthly rate.")
        result = simulate_retention_offer(customer, discount, bundle)
        st.metric("New monthly charge", f"₹{result['new_monthly_charge']:,.2f}", f"−₹{result['monthly_savings']:,.2f} / month")
        st.metric(f"Estimated {duration}-month offer cost", f"₹{result['monthly_savings'] * duration:,.2f}")
    with c2:
        st.subheader("Predicted churn probability")
        a, b = st.columns(2)
        with a: st.metric("Before offer", f"{result['current_probability']:.1%}")
        with b: st.metric("With offer", f"{result['offer_probability']:.1%}", f"{result['risk_reduction_pp']:+.1f} pp", delta_color="inverse")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=["Churn risk"], y=[result["current_probability"] * 100], marker_color=RED, text=[f"{result['current_probability']:.1%}"], textposition="outside"))
        fig.add_trace(go.Bar(name="With offer", x=["Churn risk"], y=[result["offer_probability"] * 100], marker_color=TEAL, text=[f"{result['offer_probability']:.1%}"], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_title="Predicted probability", yaxis_ticksuffix="%", yaxis_range=[0, 110], title="Before and after retention offer", legend=dict(orientation="h", y=-.2), margin=dict(l=16,r=16,t=52,b=24), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", height=365)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.info("What-if scores represent the model response to the lower monthly charge while keeping the customer's other profile fields fixed. They are directional estimates, not a causal estimate of offer effectiveness.")

