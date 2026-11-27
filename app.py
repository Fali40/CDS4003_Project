import streamlit as st
import joblib
import pandas as pd

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="E-Commerce Spend Predictor",
    page_icon="🛍️",
    layout="wide"
)

# --- 2. CUSTOM BEIGE THEME (CSS) ---
st.markdown(
    """
    <style>
    /* Main background color (Light Beige) */
    .stApp {
        background-color: #FAF9F6; 
    }
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #F0EFE6;
    }
    /* Form card styling (White background, rounded corners, shadow) */
    [data-testid="stForm"] {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
    }
    /* Darker Beige Title Banner */
    .title-banner {
        background-color: #E8E4D9; /* Darker Beige */
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.08);
        border: 1px solid #DCD6C8;
    }
    .title-banner h1 {
        color: #333333; /* Dark gray for readability */
        margin: 0;
        font-size: 2.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. LOAD MODEL AND PREPROCESSOR ---
@st.cache_resource
def load_model():
    return joblib.load('ecommerce_spend_model.pkl')

model = load_model()
ct = model.named_steps['preprocessor']
num_cols = list(ct.transformers_[0][2])
cat_cols = list(ct.transformers_[1][2])
ohe = ct.named_transformers_['cat'].named_steps['onehot']

engineered_features = ['avg_spend_per_transaction', 'cart_conversion_rate', 'credit_utilization_ratio']
input_num_cols = [col for col in num_cols if col not in engineered_features]

# --- Helper function to set logical step sizes ---
def get_step_size(col_name):
    if col_name in ['average_income', 'lifetime_spend', 'credit_limit', 'avg_order_value']:
        return 100.0 
    elif col_name in ['age', 'num_transactions', 'num_sessions_last_30d', 'avg_cart_additions', 'avg_pages_viewed']:
        return 1.0 
    else:
        return 0.01 

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict Customer Spend"])

# ==========================================
# PAGE 1: HOME PAGE
# ==========================================
if page == "Home":
    # Custom Title Banner
    st.markdown("""
        <div class="title-banner">
            <h1>Welcome to the Global E-Commerce Spend Predictor 🛍️</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    This application uses Machine Learning to predict **Future Customer Spend (90-Day)** based on a customer's historical demographics, behavior, and financial data. 
    
    ### How it works
    *   **Dataset:** Trained on a simulated global e-commerce dataset.
    *   **Model:** We use a tuned **Random Forest Regressor** wrapped in a data preprocessing pipeline.
    *   **Feature Engineering:** The app automatically calculates derived metrics like *Average Spend per Transaction* and *Cart Conversion Rates* behind the scenes.
    
    ### How to use this app:
    1. Click on **"Predict Customer Spend"** in the sidebar.
    2. Fill in the customer's details.
    3. Click **Predict** to see the estimated 90-day spend.
    """)
    st.info("👈 Use the sidebar on the left to get started.")

# ==========================================
# PAGE 2: PREDICTION PAGE
# ==========================================
elif page == "Predict Customer Spend":
    # Custom Title Banner
    st.markdown("""
        <div class="title-banner">
            <h1>Customer Spend Prediction</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Result Placeholder
    result_placeholder = st.empty()
    
    # Add a tip for the user
    st.caption("💡 **Tip:** To enter numbers quickly (like Income or Spend), just click directly on the number box and type your value instead of using the +/- buttons.")
    
    defaults = {
        'age': 35.0, 'loyalty_score': 0.5, 'discount_usage_rate': 0.25, 
        'email_open_rate': 0.3, 'avg_review_score': 4.0, 'credit_limit': 5000.0,
        'avg_order_value': 150.0, 'lifetime_spend': 2000.0, 'avg_session_duration': 5.0,
        'avg_cart_additions': 2.0, 'avg_pages_viewed': 10.0, 'bounce_rate': 0.4,
        'num_sessions_last_30d': 5.0, 'num_transactions': 10.0, 'average_income': 50000.0,
        'urban_ratio': 0.5, 'internet_penetration': 0.8, 'is_active': 1.0
    }

    user_inputs = {}
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        mid_num = len(input_num_cols) // 2
        col1_num = input_num_cols[:mid_num]
        col2_num = input_num_cols[mid_num:]
        
        mid_cat = len(cat_cols) // 2
        col1_cat = cat_cols[:mid_cat]
        col2_cat = cat_cols[mid_cat:]

        # --- LEFT COLUMN ---
        with col1:
            st.subheader("👤 Customer Demographics")
            for col in col1_num:
                user_inputs[col] = st.number_input(
                    label=f"{col.replace('_', ' ').title()}",
                    value=float(defaults.get(col, 0.0)),
                    step=get_step_size(col)
                )
            for col in col1_cat:
                cats = list(ohe.categories_[cat_cols.index(col)]) if hasattr(ohe, 'categories_') else ['other']
                user_inputs[col] = st.selectbox(
                    label=f"{col.replace('_', ' ').title()}", 
                    options=cats
                )

        # --- RIGHT COLUMN ---
        with col2:
            st.subheader("📈 Shopping Behavior")
            for col in col2_num:
                user_inputs[col] = st.number_input(
                    label=f"{col.replace('_', ' ').title()}",
                    value=float(defaults.get(col, 0.0)),
                    step=get_step_size(col)
                )
            for col in col2_cat:
                cats = list(ohe.categories_[cat_cols.index(col)]) if hasattr(ohe, 'categories_') else ['other']
                user_inputs[col] = st.selectbox(
                    label=f"{col.replace('_', ' ').title()}", 
                    options=cats
                )

        st.divider()
        submit_button = st.form_submit_button(label="🔮 Predict Future Spend", type="primary", use_container_width=True)

    # Handle Prediction
    if submit_button:
        with st.spinner("Calculating prediction..."):
            input_df = pd.DataFrame([user_inputs])
            
            input_df['avg_spend_per_transaction'] = input_df['lifetime_spend'] / (input_df['num_transactions'] + 1)
            input_df['cart_conversion_rate'] = input_df['avg_cart_additions'] / (input_df['avg_pages_viewed'] + 1)
            input_df['credit_utilization_ratio'] = input_df['lifetime_spend'] / (input_df['credit_limit'] + 1)
            
            prediction = model.predict(input_df)[0]
            
            with result_placeholder.container():
                st.success(f"### 🎯 Predicted 90-Day Spend: **${prediction:,.2f}**")
                st.divider()