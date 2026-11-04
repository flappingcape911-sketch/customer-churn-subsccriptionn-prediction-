"""Interactive Plotly charts used by the Streamlit dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#00A6A6"
GOLD = "#F2A541"
RED = "#D9534F"
PALETTE = [NAVY, TEAL, GOLD, BLUE, RED, "#6B8EAD"]
PLOT_LAYOUT = dict(margin=dict(l=16, r=16, t=52, b=16), paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Aptos, Segoe UI, sans-serif", color="#34445A"),
                   title_font=dict(size=16, color=NAVY), legend_title_text="")


def _finish(fig, height=330):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E7ECF2", zeroline=False)
    return fig


def churn_donut(frame: pd.DataFrame):
    counts = frame["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig = px.pie(values=counts.values, names=["Retained", "Churned"], hole=.68,
                 color_discrete_sequence=[TEAL, RED], title="Customer mix")
    fig.update_traces(textposition="outside", textinfo="label+percent", marker=dict(line=dict(color="white", width=3)))
    return _finish(fig)


def contract_churn_bar(frame: pd.DataFrame):
    rates = frame.assign(_churn=frame["Churn"].eq("Yes").astype(int)).groupby("Contract", observed=False)["_churn"].mean().mul(100).reset_index(name="Churn rate")
    order = [x for x in ["Month-to-month", "One year", "Two year"] if x in rates["Contract"].values]
    fig = px.bar(rates, x="Contract", y="Churn rate", color="Contract", category_orders={"Contract": order},
                 color_discrete_sequence=[RED, GOLD, TEAL], text_auto=".1f", title="Churn rate by contract")
    fig.update_traces(texttemplate="%{text}%", textposition="outside", showlegend=False, cliponaxis=False)
    fig.update_yaxes(title="Churn rate (%)", ticksuffix="%", rangemode="tozero")
    return _finish(fig)


def tenure_risk_trend(frame: pd.DataFrame):
    data = frame.assign(_churn=frame["Churn"].eq("Yes").astype(int)).groupby("tenure", as_index=False)["_churn"].mean().rename(columns={"_churn": "Average churn risk"})
    data["Average churn risk"] *= 100
    fig = px.area(data, x="tenure", y="Average churn risk", title="Churn risk across tenure",
                  color_discrete_sequence=[BLUE])
    fig.add_scatter(x=data["tenure"], y=data["Average churn risk"], mode="lines",
                    line=dict(color=TEAL, width=2), name="Risk trend", showlegend=False)
    fig.update_xaxes(title="Tenure (months)")
    fig.update_yaxes(title="Average churn risk", ticksuffix="%")
    return _finish(fig)


def charges_histogram(frame: pd.DataFrame):
    fig = px.histogram(frame, x="MonthlyCharges", color="Churn", barmode="overlay", opacity=.72,
                       nbins=28, color_discrete_map={"No": TEAL, "Yes": RED},
                       labels={"MonthlyCharges": "Monthly charges (₹)", "Churn": "Churn status"},
                       title="Monthly charges by churn status")
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(tickprefix="₹")
    return _finish(fig)


def feature_importance_chart(model_bundle: dict, top_n: int = 12):
    data = model_bundle["feature_importance"].head(top_n).sort_values("Importance", ascending=True)
    fig = px.bar(data, x="Importance", y="Feature", orientation="h", title="XGBoost feature importance",
                 color="Importance", color_continuous_scale=["#B8D7E8", BLUE, NAVY])
    fig.update_layout(coloraxis_showscale=False)
    return _finish(fig, max(340, 32 * len(data) + 80))


def roc_auc_chart(model_bundle: dict):
    fig = go.Figure()
    for name, curve in model_bundle["roc_curves"].items():
        fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines",
                                 name=f"{name} · AUC {curve['auc']:.3f}", line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                             line=dict(color="#AAB5C1", width=1.5, dash="dash")))
    fig.update_layout(title="ROC-AUC model comparison", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", colorway=PALETTE, legend=dict(orientation="h", y=-.22))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _finish(fig, 390)

