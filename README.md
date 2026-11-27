# CDS4003_Project
We will be using this dataset "Global E-Commerce Extreme Regression Dataset" to predict the "future_customer_spend_90d" which is A continuous regression target representing simulated customer spending over the next 90 days.

# Global E-Commerce 90-Day Customer Spend Prediction (`CDS4003_Project`)

## Project Overview
This project builds an end-to-end machine learning regression pipeline and interactive **Streamlit** web application to predict **`future_customer_spend_90d`**—simulated customer spending over the next 90 days—using the **Global E-Commerce Extreme Regression Dataset**.

## Dataset & Modeling Challenges
- **Dataset:** `global_ecommerce_extreme_regression_dataset.csv` (~350,000 customer records) and its cleaned subset (`cleaned_global_ecommerce_dataset.csv`).
- **Target Variable:** `future_customer_spend_90d` exhibits an extreme heavy-tailed distribution with high variance and zero-inflated spending behavior.
- **Computational Optimization:** Because training and grid-searching across all ~350k rows is computationally expensive, model training and hyperparameter tuning (`GridSearchCV`) were performed on a representative **20,000-row sample** with constrained tree depth (`max_depth`) and estimator counts (`n_estimators`).
- **Evaluation Context:** Due to the extreme heavy-tailed variance of 90-day e-commerce spend, low or negative $R^2$ scores are expected and analyzed alongside error metrics (MAE / RMSE) in `Code.ipynb`.

## Methodology & Pipeline
1. **Data Cleaning & Exploratory Data Analysis (`Code.ipynb`):** Handled missing values, inspected feature distributions, and generated target distribution visualizations (`results/figures/`).
2. **Preprocessing Pipeline:** Built a `scikit-learn` `ColumnTransformer` applying numerical scaling/imputation and `OneHotEncoder` for categorical variables to prevent data leakage across train/test splits.
3. **Model Tuning & Serialization:** Tuned a `RandomForestRegressor` pipeline via `GridSearchCV` and saved the best estimator (`grid_search.best_estimator_`) to `ecommerce_spend_model.pkl` using `joblib`.
4. **Deployment (`app.py`):** Built a Streamlit interface that dynamically extracts numerical and categorical feature schemas from the saved pipeline to generate live 90-day spend predictions.

## Repository Structure
```text
CDS4003_Project/
├── Datasets/                      # Original and cleaned dataset references
├── results/
│   └── figures/                   # EDA and target distribution plots
├── Code.ipynb                     # Data preprocessing, EDA, GridSearchCV tuning, and evaluation
├── app.py                         # Streamlit web application for live inference
├── ecommerce_spend_model.pkl      # Serialized scikit-learn preprocessing + Random Forest pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
