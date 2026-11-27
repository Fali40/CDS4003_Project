import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Global E-Commerce Spend Prediction (90-Day)",
    layout="wide",
)

model = joblib.load("ecommerce_spend_model.pkl")

ct = model.named_steps["preprocessor"]
num_cols = list(ct.transformers_[0][2])
cat_cols = list(ct.transformers_[1][2])
ohe = ct.named_transformers_["cat"].named_steps["onehot"]

DEFAULT_NUMERICALS = {
    "age": 30.0,
    "loyalty_score": 0.5,
    "discount_usage_rate": 0.2,
    "email_open_rate": 0.25,
    "avg_review_score": 4.2,
    "credit_limit": 2500.0,
}


def format_label(col_name: str) -> str:
    return col_name.replace("_", " ").title()


def predict_spend(inputs_dict: dict) -> float:
    input_data = pd.DataFrame([inputs_dict])
    prediction = model.predict(input_data)
    return float(prediction[0])


st.title("Global E-Commerce Spend Prediction (90-Day)")
st.caption("Predicts `future_customer_spend_90d` using the trained Random Forest pipeline.")

user_inputs = {}

if num_cols:
    st.subheader("Numerical Features")
    col1, col2 = st.columns(2)
    for idx, col in enumerate(num_cols):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if col == "age":
                user_inputs[col] = st.number_input(
                    format_label(col),
                    min_value=0,
                    max_value=120,
                    value=int(DEFAULT_NUMERICALS.get(col, 30)),
                    step=1,
                )
            else:
                default_val = float(DEFAULT_NUMERICALS.get(col, 0.0))
                user_inputs[col] = st.number_input(
                    format_label(col),
                    value=default_val,
                    step=0.01 if abs(default_val) <= 1.0 else 10.0,
                )

if cat_cols:
    st.subheader("Categorical Features")
    c1, c2 = st.columns(2)
    for idx, col in enumerate(cat_cols):
        cats = list(ohe.categories_[idx]) if hasattr(ohe, "categories_") else ["other"]
        options = cats if len(cats) > 0 else ["other"]
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            user_inputs[col] = st.selectbox(format_label(col), options)

st.divider()

if st.button("Predict", type="primary"):
    result = predict_spend(user_inputs)
    st.success(f"Predicted `future_customer_spend_90d`: **{result:,.2f}**")