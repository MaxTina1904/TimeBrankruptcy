import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Time-to-Bankruptcy Predictor",
    page_icon="📉",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📉 Time-to-Bankruptcy Predictor")
st.markdown(
    "Enter a company's financial ratios below to predict how many years "
    "until potential bankruptcy filing. Based on NYSE/NASDAQ data (1999–2018)."
)
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("📊 Liquidity Ratios")
col1, col2, col3 = st.columns(3)
with col1:
    current_ratio = st.number_input(
        "Current Ratio", value=1.5, min_value=0.0, max_value=20.0, step=0.1,
        help="Current Assets / Current Liabilities"
    )
with col2:
    quick_ratio = st.number_input(
        "Quick Ratio", value=1.0, min_value=0.0, max_value=20.0, step=0.1,
        help="(Current Assets − Inventory) / Current Liabilities"
    )
with col3:
    cash_ratio = st.number_input(
        "Cash Ratio", value=0.5, min_value=0.0, max_value=10.0, step=0.1,
        help="Cash & Equivalents / Current Liabilities"
    )

st.subheader("📈 Profitability Ratios")
col4, col5, col6 = st.columns(3)
with col4:
    roa = st.number_input(
        "ROA", value=0.05, min_value=-1.0, max_value=1.0, step=0.01,
        help="Net Income / Total Assets"
    )
with col5:
    gross_margin = st.number_input(
        "Gross Margin", value=0.30, min_value=-1.0, max_value=1.0, step=0.01,
        help="(Revenue − COGS) / Revenue"
    )
with col6:
    ebit_margin = st.number_input(
        "EBIT Margin", value=0.08, min_value=-1.0, max_value=1.0, step=0.01,
        help="EBIT / Total Revenue"
    )

st.subheader("⚙️ Efficiency & Leverage Ratios")
col7, col8, col9 = st.columns(3)
with col7:
    asset_turnover = st.number_input(
        "Asset Turnover", value=0.8, min_value=0.0, max_value=5.0, step=0.01,
        help="Total Revenue / Total Assets"
    )
with col8:
    debt_to_equity = st.number_input(
        "Debt-to-Equity", value=1.5, min_value=-20.0, max_value=50.0, step=0.1,
        help="Total Liabilities / Shareholders' Equity"
    )
with col9:
    leverage_ratio = st.number_input(
        "Leverage Ratio", value=0.5, min_value=0.0, max_value=1.0, step=0.01,
        help="Total Liabilities / Total Assets"
    )

st.subheader("🏦 Balance Sheet Ratios")
col10, col11, col12 = st.columns(3)
with col10:
    working_capital_ratio = st.number_input(
        "Working Capital Ratio", value=0.1, min_value=-1.0, max_value=1.0, step=0.01,
        help="(Current Assets − Current Liabilities) / Total Assets"
    )
with col11:
    retained_earnings_ratio = st.number_input(
        "Retained Earnings Ratio", value=0.1, min_value=-2.0, max_value=2.0, step=0.01,
        help="Retained Earnings / Total Assets"
    )
with col12:
    receivables_ratio = st.number_input(
        "Receivables Ratio", value=0.15, min_value=0.0, max_value=2.0, step=0.01,
        help="Total Receivables / Total Revenue"
    )

st.subheader("🧮 Altman Z-Score Proxy")
altman_z_proxy = st.number_input(
    "Altman Z-Proxy", value=2.0, min_value=-10.0, max_value=20.0, step=0.1,
    help="1.2×WC_ratio + 1.4×RE_ratio + 3.3×ROA + 0.6×(Equity/Liabilities) + Asset_Turnover"
)

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Years to Bankruptcy", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([[
        current_ratio, quick_ratio, roa, asset_turnover, gross_margin,
        ebit_margin, debt_to_equity, working_capital_ratio,
        retained_earnings_ratio, receivables_ratio, leverage_ratio,
        cash_ratio, altman_z_proxy
    ]], columns=[
        'current_ratio', 'quick_ratio', 'roa', 'asset_turnover', 'gross_margin',
        'ebit_margin', 'debt_to_equity', 'working_capital_ratio',
        'retained_earnings_ratio', 'receivables_ratio', 'leverage_ratio',
        'cash_ratio', 'altman_z_proxy'
    ])

    prediction = float(np.clip(model.predict(input_df)[0], 0, 20))

    # ── Risk classification ───────────────────────────────────────────────────
    if prediction <= 2:
        risk_label, risk_color, risk_icon = "High Risk", "red", "🔴"
    elif prediction <= 5:
        risk_label, risk_color, risk_icon = "Moderate Risk", "orange", "🟠"
    else:
        risk_label, risk_color, risk_icon = "Low Risk", "green", "🟢"

    st.markdown(f"""
    <div style="
        background: {'#ffe0e0' if risk_color=='red' else '#fff3e0' if risk_color=='orange' else '#e0f4e0'};
        border-left: 5px solid {'#e53935' if risk_color=='red' else '#fb8c00' if risk_color=='orange' else '#43a047'};
        padding: 1.2rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    ">
        <h3 style="margin:0; color: {'#b71c1c' if risk_color=='red' else '#e65100' if risk_color=='orange' else '#1b5e20'}">
            {risk_icon} {prediction:.1f} years until predicted bankruptcy
        </h3>
        <p style="margin: 0.4rem 0 0 0; font-size: 1rem; color: #444;">
            Risk classification: <strong>{risk_label}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input summary table ───────────────────────────────────────────────────
    st.markdown("#### Input Summary")
    summary = input_df.T.reset_index()
    summary.columns = ["Financial Ratio", "Value"]
    summary["Value"] = summary["Value"].round(4)
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Regression pipeline (StandardScaler + Ridge) · "
    "Dataset: American Bankruptcy Prediction — NYSE/NASDAQ 1999–2018 · "
    "Metric: Years until bankruptcy filing (right-censored survival target)"
)
