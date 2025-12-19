"""
Medicine Sales BI Dashboard (Streamlit)

Covers BI-hw2 requirements:
- Descriptive analytics: Sales + (estimated) Profit by medicine type/subtype, time, customer profile (country)
- Peak/low moments for a selected drug type (max/min sales)
- Forecast next period(s) using trend + seasonality
- Price-change scenario analysis using estimated price elasticity
- Exports to Excel (filtered data + analytics + predictions)
- Map charts (choropleth by country)

How to run:
1) pip install -r requirements.txt
2) streamlit run app.py
3) Put your CSV in ./data/medicine_sales.csv or upload it in the app
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression


# -------------------------
# Config
# -------------------------

st.set_page_config(page_title="Medicine Sales BI (HW2)", layout="wide")


# -------------------------
# Data model helpers
# -------------------------

@dataclass(frozen=True)
class ForecastResult:
    """Container for forecast outputs used by both UI and Excel export."""
    history: pd.DataFrame  # columns: ds, y
    forecast: pd.DataFrame  # columns: ds, yhat
    model_name: str


# -------------------------
# Loading & preprocessing
# -------------------------

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file, default_path: str = "data/medicine_sales.csv") -> pd.DataFrame:
    """Load a CSV either from Streamlit uploader or from a local default path."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(default_path)


def add_derived_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns + derive Type/Subtype, revenue, unit_price, and normalized country names."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Expected source columns (based on your dataset)
    # Date, Product, Sales Person, Boxes Shipped, Amount ($), Country
    df["date"] = pd.to_datetime(df.get("Date"), errors="coerce")

    df["product"] = df.get("Product", "").astype(str).str.strip()
    df["sales_person"] = df.get("Sales Person", "").astype(str).str.strip()
    df["country_raw"] = df.get("Country", "").astype(str).str.strip()

    df["boxes"] = pd.to_numeric(df.get("Boxes Shipped"), errors="coerce").fillna(0).astype(float)
    df["revenue"] = pd.to_numeric(df.get("Amount ($)"), errors="coerce").fillna(0).astype(float)

    # Type/Subtype parsing: subtype = last word, type = rest
    df["subtype"] = df["product"].str.split().str[-1]
    df["type"] = df["product"].str.replace(r"\s+\S+$", "", regex=True)

    # Unit price inferred from revenue/boxes
    df["unit_price"] = np.where(df["boxes"] > 0, df["revenue"] / df["boxes"], np.nan)

    # Normalize country names for plotly maps
    country_map = {"UK": "United Kingdom", "USA": "United States"}
    df["country"] = df["country_raw"].replace(country_map)

    return df


def safe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Create a copy of df with SQL-friendly column names and return (df_safe, mapping)."""
    mapping: Dict[str, str] = {}
    df2 = df.copy()
    for c in df2.columns:
        safe = (
            c.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("$", "usd")
            .replace("-", "_")
        )
        safe = "".join(ch for ch in safe if ch.isalnum() or ch == "_")
        if not safe:
            safe = "col"
        if safe in mapping.values():
            safe = f"{safe}_{abs(hash(c)) % 9999}"
        mapping[c] = safe
        df2.rename(columns={c: safe}, inplace=True)
    return df2, mapping


def apply_filters(
    df: pd.DataFrame,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    products: Optional[list],
    types_: Optional[list],
    subtypes: Optional[list],
    countries: Optional[list],
    sales_people: Optional[list],
    sql_expr: str,
) -> pd.DataFrame:
    """Apply UI filters + optional SQL-like query (pandas query) safely."""
    out = df.copy()

    start, end = date_range
    out = out[(out["date"] >= start) & (out["date"] <= end)]

    if products:
        out = out[out["product"].isin(products)]
    if types_:
        out = out[out["type"].isin(types_)]
    if subtypes:
        out = out[out["subtype"].isin(subtypes)]
    if countries:
        out = out[out["country"].isin(countries)]
    if sales_people:
        out = out[out["sales_person"].isin(sales_people)]

    # SQL-like dynamic filter using df.query on a safe-column clone
    sql_expr = (sql_expr or "").strip()
    if sql_expr:
        safe_df, _mapping = safe_columns(out)
        try:
            safe_filtered = safe_df.query(sql_expr, engine="python")
            out = out.loc[safe_filtered.index]
        except Exception as e:
            st.warning(f"SQL-like filter error: {e}")

    return out


# -------------------------
# Analytics helpers
# -------------------------

def add_profit(df: pd.DataFrame, gross_margin: float, cost_mode: str) -> pd.DataFrame:
    """Compute an estimated profit from revenue using either margin-only or cost-per-box assumption."""
    out = df.copy()

    gross_margin = float(np.clip(gross_margin, 0.0, 0.95))

    if cost_mode == "Profit = Revenue × Margin":
        out["profit"] = out["revenue"] * gross_margin
        out["cost"] = out["revenue"] - out["profit"]
    else:
        # Estimate cost-per-box from unit price and margin, then profit = revenue - cost
        cost_per_box = out["unit_price"] * (1.0 - gross_margin)
        out["cost"] = np.where(out["boxes"] > 0, cost_per_box * out["boxes"], 0.0)
        out["profit"] = out["revenue"] - out["cost"]

    return out


def group_time(df: pd.DataFrame, freq: str, dims: list, metric: str) -> pd.DataFrame:
    """Aggregate metric over time (freq) and dimensions."""
    if df.empty:
        return pd.DataFrame(columns=["period", metric, *dims])

    g = (
        df.dropna(subset=["date"])
        .set_index("date")
        .groupby([pd.Grouper(freq=freq), *dims])[metric]
        .sum()
        .reset_index()
        .rename(columns={"date": "period"})
    )
    g["period"] = pd.to_datetime(g["period"])
    return g


def max_min_moments(df: pd.DataFrame, freq: str, type_value: str, metric: str) -> pd.DataFrame:
    """Return the max/min period for a selected type."""
    if df.empty or not type_value:
        return pd.DataFrame(columns=["type", "moment", "period", metric])

    s = (
        df[df["type"] == type_value]
        .dropna(subset=["date"])
        .set_index("date")
        .resample(freq)[metric]
        .sum()
        .asfreq(freq)
        .fillna(0.0)
    )
    if s.empty:
        return pd.DataFrame(columns=["type", "moment", "period", metric])

    max_period = s.idxmax()
    min_period = s.idxmin()

    return pd.DataFrame(
        [
            {"type": type_value, "moment": "MAX", "period": max_period, metric: float(s.loc[max_period])},
            {"type": type_value, "moment": "MIN", "period": min_period, metric: float(s.loc[min_period])},
        ]
    )


# -------------------------
# Forecasting + elasticity
# -------------------------

def build_linear_seasonal_features(dates: pd.Series, t0: int = 0) -> pd.DataFrame:
    """Create trend + simple seasonality features for a linear regression forecaster."""
    X = pd.DataFrame({"ds": pd.to_datetime(dates)})
    X["t"] = np.arange(t0, t0 + len(X), dtype=float)
    X["month"] = X["ds"].dt.month
    X["dow"] = X["ds"].dt.dayofweek
    X = pd.get_dummies(X[["t", "month", "dow"]], columns=["month", "dow"], drop_first=True)
    return X


def fit_forecast(df: pd.DataFrame, entity_col: str, entity_value: str, metric: str, freq: str, horizon: int) -> ForecastResult:
    """Fit a lightweight trend+seasonality model and forecast next horizon periods."""
    data = df[df[entity_col] == entity_value].copy()
    data = data.dropna(subset=["date"])
    if data.empty:
        return ForecastResult(history=pd.DataFrame(columns=["ds", "y"]), forecast=pd.DataFrame(columns=["ds", "yhat"]), model_name="LinearSeasonal")

    y = (
        data.set_index("date")
        .sort_index()
        .resample(freq)[metric]
        .sum()
        .asfreq(freq)
        .fillna(0.0)
    )

    hist = pd.DataFrame({"ds": y.index, "y": y.values})

    X = build_linear_seasonal_features(hist["ds"], t0=0)
    model = LinearRegression()
    model.fit(X, hist["y"].values)

    last = hist["ds"].iloc[-1]
    future_dates = pd.date_range(start=last, periods=horizon + 1, freq=freq, inclusive="right")
    Xf = build_linear_seasonal_features(pd.Series(future_dates), t0=len(hist))
    Xf = Xf.reindex(columns=X.columns, fill_value=0)

    yhat = model.predict(Xf)
    yhat = np.maximum(yhat, 0.0)

    fc = pd.DataFrame({"ds": future_dates, "yhat": yhat})
    return ForecastResult(history=hist, forecast=fc, model_name="LinearSeasonal")


def estimate_elasticity(df: pd.DataFrame, min_obs: int = 10) -> pd.DataFrame:
    """Estimate constant price elasticity per product from (log boxes) ~ (log unit_price)."""
    rows = []
    for prod, g in df.groupby("product"):
        gg = g.dropna(subset=["unit_price", "boxes"])
        gg = gg[(gg["unit_price"] > 0) & (gg["boxes"] > 0)]
        if len(gg) < min_obs:
            rows.append({"product": prod, "elasticity": np.nan, "n_obs": len(gg)})
            continue
        X = np.log(gg[["unit_price"]].values)
        y = np.log(gg["boxes"].values)
        m = LinearRegression()
        m.fit(X, y)
        rows.append({"product": prod, "elasticity": float(m.coef_[0]), "n_obs": len(gg)})
    return pd.DataFrame(rows).sort_values("product")


def price_scenario(
    base_forecast_boxes: pd.DataFrame,
    base_unit_price: float,
    elasticity: float,
    price_change_pct: float,
    gross_margin: float,
) -> pd.DataFrame:
    """Apply a constant-elasticity price change to forecasted boxes and compute revenue/profit."""
    out = base_forecast_boxes.copy()
    out = out.rename(columns={"yhat": "boxes_baseline"})
    out["price_baseline"] = base_unit_price

    price_multiplier = 1.0 + (price_change_pct / 100.0)
    out["price_scenario"] = out["price_baseline"] * price_multiplier

    if np.isfinite(elasticity):
        out["boxes_scenario"] = out["boxes_baseline"] * (price_multiplier ** elasticity)
    else:
        out["boxes_scenario"] = out["boxes_baseline"]

    out["revenue_baseline"] = out["boxes_baseline"] * out["price_baseline"]
    out["revenue_scenario"] = out["boxes_scenario"] * out["price_scenario"]

    cost_per_box = out["price_baseline"] * (1.0 - gross_margin)
    out["profit_baseline"] = out["revenue_baseline"] - (cost_per_box * out["boxes_baseline"])
    out["profit_scenario"] = out["revenue_scenario"] - (cost_per_box * out["boxes_scenario"])
    return out


def excel_bytes(
    filtered: pd.DataFrame,
    summary: pd.DataFrame,
    maxmin: pd.DataFrame,
    forecast: Optional[ForecastResult],
    scenario: Optional[pd.DataFrame],
    elasticity: pd.DataFrame,
) -> bytes:
    """Create a multi-sheet Excel file for 'pred.&desc. analytics' export."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="FilteredData")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        maxmin.to_excel(writer, index=False, sheet_name="MaxMin")
        elasticity.to_excel(writer, index=False, sheet_name="Elasticity")

        if forecast is not None:
            forecast.history.to_excel(writer, index=False, sheet_name="Forecast_History")
            forecast.forecast.to_excel(writer, index=False, sheet_name="Forecast_Next")

        if scenario is not None:
            scenario.to_excel(writer, index=False, sheet_name="PriceScenario")
    return bio.getvalue()


# -------------------------
# UI
# -------------------------

st.title("💊 Medicine Sales BI Dashboard (HW2)")

with st.expander("Dataset input", expanded=True):
    uploaded = st.file_uploader("Upload CSV (or leave empty to use ./data/medicine_sales.csv)", type=["csv"])
    st.caption("Expected columns: Date, Product, Sales Person, Boxes Shipped, Amount ($), Country")

try:
    raw = load_csv(uploaded_file=uploaded)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

df = add_derived_columns(raw)

df_non_null_dates = df.dropna(subset=["date"])
if df_non_null_dates.empty:
    st.error("No valid dates found in the dataset.")
    st.stop()

min_date = df_non_null_dates["date"].min()
max_date = df_non_null_dates["date"].max()

st.sidebar.header("Filters (Interactivity)")
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

all_types = sorted(df["type"].dropna().unique().tolist())
all_subtypes = sorted(df["subtype"].dropna().unique().tolist())
all_products = sorted(df["product"].dropna().unique().tolist())
all_countries = sorted(df["country"].dropna().unique().tolist())
all_sales_people = sorted(df["sales_person"].dropna().unique().tolist())

types_sel = st.sidebar.multiselect("Type", options=all_types, default=all_types)
subtypes_sel = st.sidebar.multiselect("Subtype", options=all_subtypes, default=all_subtypes)
products_sel = st.sidebar.multiselect("Product", options=all_products, default=all_products)
countries_sel = st.sidebar.multiselect("Customer profile (Country)", options=all_countries, default=all_countries)
sales_sel = st.sidebar.multiselect("Sales person", options=all_sales_people, default=all_sales_people)

sql_help = st.sidebar.expander("SQL-like filter (pandas query)")
with sql_help:
    st.write("Use safe column names (lowercase, underscores). Example:")
    st.code("country == 'United Kingdom' and revenue > 300 and product.str.contains('Spray')")
    safe_df_tmp, mapping_tmp = safe_columns(df)
    st.write("Safe column mapping:")
    st.json(mapping_tmp)

sql_expr = st.sidebar.text_input("SQL-like expression (optional)", value="")

st.sidebar.header("Profit assumptions")
gross_margin = st.sidebar.slider("Gross margin", min_value=0.0, max_value=0.9, value=0.35, step=0.01)
cost_mode = st.sidebar.radio("Profit model", ["Profit = Revenue × Margin", "Profit = Revenue − CostPerBox"], index=0)

filtered = apply_filters(
    df=df,
    date_range=(start_date, end_date),
    products=products_sel,
    types_=types_sel,
    subtypes=subtypes_sel,
    countries=countries_sel,
    sales_people=sales_sel,
    sql_expr=sql_expr,
)
filtered = add_profit(filtered, gross_margin=gross_margin, cost_mode=cost_mode)

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Revenue ($)", f"{filtered['revenue'].sum():,.2f}")
kpi2.metric("Boxes shipped", f"{filtered['boxes'].sum():,.0f}")
kpi3.metric("Estimated profit ($)", f"{filtered['profit'].sum():,.2f}")
kpi4.metric("Avg unit price ($/box)", f"{filtered['unit_price'].dropna().mean():,.2f}")

st.header("1) Sales/Profit by Type/Subtype x Time x Customer profile")

left, right = st.columns([1, 1])

with left:
    metric = st.selectbox("Metric", ["revenue", "profit", "boxes"], index=0)
    freq_label = st.selectbox("Time aggregation", ["Daily", "Weekly", "Monthly"], index=2)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq_label]
    dims = st.multiselect("Break down by", ["type", "subtype", "product", "country", "sales_person"], default=["type", "country"])

    g = group_time(filtered, freq=freq, dims=dims, metric=metric)

    if not g.empty:
        if dims:
        
            g["series"] = g[dims].astype(str).agg(" | ".join, axis=1)

            fig = px.line(
                g,
                x="period",
                y=metric,
                color="series",
                markers=True,
                hover_data=dims,
            )
        else:
            fig = px.line(g, x="period", y=metric, markers=True)

        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Pivot summary (descriptive)")
    pivot_dims = st.multiselect("Pivot rows", ["type", "subtype", "product", "country", "sales_person"], default=["type", "subtype"])
    if pivot_dims:
        summary = (
            filtered.groupby(pivot_dims)[["revenue", "profit", "boxes"]]
            .sum()
            .sort_values("revenue", ascending=False)
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            [{"revenue": filtered["revenue"].sum(), "profit": filtered["profit"].sum(), "boxes": filtered["boxes"].sum()}]
        )
    st.dataframe(summary, use_container_width=True, height=340)

st.subheader("Map: metric by country")
metric_map = st.selectbox("Map metric", ["revenue", "profit", "boxes"], index=0, key="mapmetric")
country_agg = filtered.groupby("country")[metric_map].sum().reset_index()
map_fig = px.choropleth(country_agg, locations="country", locationmode="country names", color=metric_map, hover_name="country")
st.plotly_chart(map_fig, use_container_width=True)

st.header("2) Max/Min moments for a selected drug type")
type_pick = st.selectbox("Choose Type", options=sorted(filtered["type"].unique().tolist()))
maxmin = max_min_moments(filtered, freq=freq, type_value=type_pick, metric=metric)
st.dataframe(maxmin, use_container_width=True)

st.header("3) Forecast next period(s) from trend")
fc_col1, fc_col2 = st.columns([1, 2])

with fc_col1:
    entity_level = st.selectbox("Forecast level", ["product", "type"], index=0)
    entity_options = sorted(filtered[entity_level].unique().tolist())
    entity_value = st.selectbox("Select item", options=entity_options)
    fc_metric = st.selectbox("Forecast metric", ["boxes", "revenue"], index=0)
    fc_freq_label = st.selectbox("Forecast frequency", ["Weekly", "Monthly"], index=1)
    fc_freq = {"Weekly": "W", "Monthly": "MS"}[fc_freq_label]
    horizon = st.number_input("Horizon (periods)", min_value=1, max_value=24, value=6, step=1)

    forecast_result = fit_forecast(filtered, entity_col=entity_level, entity_value=entity_value, metric=fc_metric, freq=fc_freq, horizon=int(horizon))

with fc_col2:
    if not forecast_result.history.empty:
        hist = forecast_result.history.copy()
        fc = forecast_result.forecast.copy()
        hist["series"] = "history"
        fc_plot = fc.rename(columns={"yhat": "y"}).copy()
        fc_plot["series"] = "forecast"

        both = pd.concat([hist[["ds", "y", "series"]], fc_plot[["ds", "y", "series"]]], ignore_index=True)
        fc_fig = px.line(both, x="ds", y="y", color="series")
        st.plotly_chart(fc_fig, use_container_width=True)

        st.caption(f"Model: {forecast_result.model_name}")
        st.dataframe(forecast_result.forecast, use_container_width=True, height=240)
    else:
        st.info("Not enough data to forecast for the selected filters.")

st.header("4) Sales evolution under price change (scenario)")
elast_tbl = estimate_elasticity(filtered)
prod_for_scenario = st.selectbox("Scenario product", options=sorted(filtered["product"].unique().tolist()))
base_price = float(filtered[filtered["product"] == prod_for_scenario]["unit_price"].dropna().mean() or 0.0)

row = elast_tbl[elast_tbl["product"] == prod_for_scenario]
est_e = float(row["elasticity"].iloc[0]) if not row.empty else np.nan
elasticity_override = st.number_input("Elasticity (override allowed)", value=float(est_e) if np.isfinite(est_e) else -0.5, step=0.1)

price_change = st.slider("Price change (%)", max_value=-50, min_value=50, value=10, step=1)

base_fc = fit_forecast(filtered, entity_col="product", entity_value=prod_for_scenario, metric="boxes", freq=fc_freq, horizon=int(horizon))

scenario_df = None
if not base_fc.forecast.empty and base_price > 0:
    scenario_df = price_scenario(
        base_forecast_boxes=base_fc.forecast[["ds", "yhat"]],
        base_unit_price=base_price,
        elasticity=float(elasticity_override),
        price_change_pct=float(price_change),
        gross_margin=float(gross_margin),
    )

    scen_plot = scenario_df.melt(
        id_vars=["ds"],
        value_vars=["boxes_baseline", "boxes_scenario"],
        var_name="series",
        value_name="boxes",
    )
    scen_fig = px.line(scen_plot, x="ds", y="boxes", color="series")
    st.plotly_chart(scen_fig, use_container_width=True)

    st.dataframe(scenario_df, use_container_width=True, height=280)
else:
    st.info("Scenario requires a valid base unit price and enough data for forecasting.")

with st.expander("Elasticity estimates (for transparency)"):
    st.dataframe(elast_tbl, use_container_width=True)

st.header("Export to Excel (descriptive + predictive analytics)")
export_bytes = excel_bytes(
    filtered=filtered,
    summary=summary,
    maxmin=maxmin,
    forecast=forecast_result,
    scenario=scenario_df,
    elasticity=elast_tbl,
)

st.download_button(
    label="⬇️ Download Excel report",
    data=export_bytes,
    file_name="medicine_sales_bi_hw2_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("Tip: keep your project folder with source code + dataset + exported report + presentation to satisfy C4.")
