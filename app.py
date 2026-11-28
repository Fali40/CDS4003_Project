from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Global E-Commerce Spend Predictor", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "ecommerce_spend_model.pkl"
DATA_FILE = BASE_DIR / "cleaned_global_ecommerce_dataset.csv"

# Features the app calculates itself (the user never types these)
DERIVED_FEATURES = [
    "avg_spend_per_transaction",
    "cart_conversion_rate",
    "credit_utilization_ratio",
]

# Left column of the form; every other numeric feature goes in the right column
DEMOGRAPHIC_FEATURES = [
    "age", "loyalty_score", "discount_usage_rate", "email_open_rate",
    "avg_review_score", "credit_limit", "is_active", "average_income",
    "urban_ratio", "internet_penetration",
]

# Starting values shown in the form (anything not listed starts at 0)
DEFAULTS = {
    "age": 35.0, "loyalty_score": 0.5, "discount_usage_rate": 0.25,
    "email_open_rate": 0.30, "avg_review_score": 4.0, "credit_limit": 2500.0,
    "is_active": 1.0, "avg_order_value": 100.0, "lifetime_spend": 1000.0,
    "avg_session_duration": 5.0, "avg_cart_additions": 2.0,
    "avg_pages_viewed": 10.0, "bounce_rate": 0.40,
    "num_sessions_last_30d": 5.0, "num_transactions": 5.0,
    "average_income": 30000.0, "urban_ratio": 0.5, "internet_penetration": 0.7,
}

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {background-color: #eeede4;}
    .banner {background: #e8e4da; padding: 2rem; border-radius: 16px;
             text-align: center; margin-bottom: 1.5rem;
             box-shadow: 0 2px 6px rgba(0,0,0,0.08);}
    .banner h1 {margin: 0; font-size: 2.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def banner(title: str):
    st.markdown(f'<div class="banner"><h1>{title}</h1></div>', unsafe_allow_html=True)


def label(name: str) -> str:
    return name.replace("_", " ").title()


# ------------------------------------------------------------------
# Load the model and read its expected features
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model file: {MODEL_FILE.name}")
    st.exception(e)
    st.stop()

preprocessor = model.named_steps["preprocessor"]

# Features come from the trained pipeline, so the form always matches the model
num_cols, cat_cols, cat_options = [], [], {}
for name, transformer, columns in preprocessor.transformers_:
    if name == "num":
        num_cols = list(columns)
    elif name == "cat":
        cat_cols = list(columns)
        ohe = transformer.named_steps["onehot"]
        for col, cats in zip(columns, ohe.categories_):
            cat_options[col] = [str(c) for c in cats]


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
def predict_spend(inputs: dict) -> float:
    data = dict(inputs)

    # Derived features (same formulas used in the notebook)
    data["avg_spend_per_transaction"] = data.get("lifetime_spend", 0.0) / (data.get("num_transactions", 0.0) + 1)
    data["cart_conversion_rate"] = data.get("avg_cart_additions", 0.0) / (data.get("avg_pages_viewed", 0.0) + 1)
    data["credit_utilization_ratio"] = data.get("lifetime_spend", 0.0) / (data.get("credit_limit", 0.0) + 1)

    # Column order must match training
    row = pd.DataFrame([data])[num_cols + cat_cols]
    return max(float(model.predict(row)[0]), 0.0)


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------
def home_page():
    banner("Welcome to the Global E-Commerce Spend Predictor 🛍️")
    st.header("Project Overview")
    st.write(
        "This application uses Machine Learning to predict **Future Customer Spend (90-Day)** "
        "based on a customer's historical demographics, behavior, and financial data."
    )
    st.header("How it works")
    st.markdown(
        "- **Model:** a scikit-learn regression pipeline (preprocessing + model).\n"
        "- **Feature Engineering:** the app automatically calculates derived metrics like "
        "*Average Spend per Transaction*, *Cart Conversion Rate* and *Credit Utilization* behind the scenes."
    )
    st.header("How to use this app:")
    st.markdown(
        '1. Click **"Predict Customer Spend"** to get a prediction.\n'
        '2. Click **"Data Insights"** to see what drives customer spending.'
    )


def predict_page():
    banner("Customer Spend Prediction")
    st.caption("💡 **Tip:** click directly on a number box and type your value instead of using the +/- buttons.")

    inputs = {}
    typed_num_cols = [c for c in num_cols if c not in DERIVED_FEATURES]
    left_cols = [c for c in typed_num_cols if c in DEMOGRAPHIC_FEATURES]
    right_cols = [c for c in typed_num_cols if c not in DEMOGRAPHIC_FEATURES]

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👤 Customer Demographics")
            for c in left_cols:
                inputs[c] = st.number_input(label(c), value=float(DEFAULTS.get(c, 0.0)), key=f"in_{c}")
        with col2:
            st.subheader("📈 Shopping Behavior")
            for c in right_cols:
                inputs[c] = st.number_input(label(c), value=float(DEFAULTS.get(c, 0.0)), key=f"in_{c}")

        if cat_cols:
            st.subheader("🌍 Customer Profile")
            cc1, cc2 = st.columns(2)
            for i, c in enumerate(cat_cols):
                with (cc1 if i % 2 == 0 else cc2):
                    inputs[c] = st.selectbox(label(c), cat_options.get(c, ["other"]), key=f"in_{c}")

    if st.button("Predict Future Customer Spend", type="primary"):
        try:
            result = predict_spend(inputs)
            st.success(f"🎯 Predicted 90-Day Spend: ${result:,.2f}")
        except Exception as e:
            st.error("The prediction could not be generated.")
            st.exception(e)


def insights_page():
    banner("Data Insights & Model Drivers")
    col1, col2 = st.columns(2)

    # ---- Top 10 drivers ----
    with col1:
        st.header("🎯 Top 10 Drivers of Future Spend")
        regressor = model.named_steps["regressor"]
        if hasattr(regressor, "regressor_"):  # model wrapped in TransformedTargetRegressor
            regressor = regressor.regressor_

        if hasattr(regressor, "feature_importances_"):
            values = regressor.feature_importances_
            axis_label = "Relative Importance"
        elif hasattr(regressor, "coef_"):
            values = np.abs(np.ravel(regressor.coef_))
            axis_label = "Relative Importance (|coefficient|)"
        else:
            values = None

        if values is None:
            st.info("This model type does not expose feature importances.")
        else:
            names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
            # Add one-hot columns back together (e.g. country_code_usa -> country_code)
            base = []
            for n in names:
                match = next((c for c in cat_cols if n.startswith(c + "_")), None)
                base.append(match if match else n)
            top = (pd.Series(values, index=base).groupby(level=0).sum()
                   .sort_values(ascending=False).head(10)[::-1])

            st.write("This chart shows which features the model relies on the most when making a prediction.")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(top.index, top.values, color=plt.cm.YlOrBr(np.linspace(0.85, 0.3, len(top))))
            ax.set_xlabel(axis_label)
            plt.tight_layout()
            st.pyplot(fig)

    # ---- Spend distribution ----
    with col2:
        st.header("📊 Distribution of Customer Spend")
        st.write("This chart shows how spending is distributed. The data is highly skewed, with a few extreme high spenders.")
        if DATA_FILE.exists():
            spend = pd.read_csv(DATA_FILE, usecols=["future_customer_spend_90d"])["future_customer_spend_90d"]
            cap = spend.quantile(0.95)
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            ax2.hist(spend[spend <= cap], bins=40, color="#e8cfb4", edgecolor="black")
            ax2.set_xlabel("Future Spend ($) (Capped at 95th Percentile)")
            ax2.set_ylabel("Number of Customers")
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info(f"Add {DATA_FILE.name} next to app.py to show this chart.")


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict Customer Spend", "Data Insights"])

if page == "Home":
    home_page()
elif page == "Predict Customer Spend":
    predict_page()
else:
    insights_page()

