import streamlit as st
import joblib
import pandas as pd

model = joblib.load('ecommerce_spend_model.pkl')

ct = model.named_steps['preprocessor']
num_cols = list(ct.transformers_[0][2])
cat_cols = list(ct.transformers_[1][2])
ohe = ct.named_transformers_['cat'].named_steps['onehot']

def predict_spend(inputs_dict):
    input_data = pd.DataFrame([inputs_dict])
    prediction = model.predict(input_data)
    return prediction[0]

st.title("Global E-Commerce Spend Prediction (90-Day)")

user_inputs = {}
for col in num_cols:
    user_inputs[col] = st.number_input(col, value=0.0)

for idx, col in enumerate(cat_cols):
    cats = list(ohe.categories_[idx]) if hasattr(ohe, 'categories_') else ['other']
    options = cats if len(cats) > 0 else ['other']
    user_inputs[col] = st.selectbox(col, options)

if st.button("Predict"):
    result = predict_spend(user_inputs)
    st.success(f"Predicted future_customer_spend_90d: {result:.2f}")
