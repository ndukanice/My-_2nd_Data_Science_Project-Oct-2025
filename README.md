# Customer Churn Prediction & CLV Analysis

Predict customer churn risk and estimate Customer Lifetime Value (CLV) using machine learning.

## Features
- **3 Models**: Logistic Regression, Random Forest, XGBoost
- **CLV Analysis**: Segment customers by value quartiles
- **Interpretability**: Feature importance for all models
- **Interactive App**: Streamlit web interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add data to `data/raw/Telco-Customer-Churn.csv`

3. Run pipeline:
```bash
python src/data_prep.py
python src/train_models.py
python src/clv_analysis.py
streamlit run app.py
```

## Project Structure
```
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Train/test splits
├── models/               # Trained pipelines
├── src/
│   ├── data_prep.py      # Data cleaning & feature engineering
│   ├── train_models.py   # Model training
│   ├── clv_analysis.py   # CLV calculations
│   ├── interpretability.py  # Feature importance
│   └── predict.py        # Single prediction logic
└── app.py                # Streamlit interface
```

## Model Performance
- **Recall ≥60%** on test set (churn detection requirement)
- **AUC ≥0.75** across all models
- Class imbalance handled via balanced weights

## CLV Assumptions
- **Expected Tenure** based on contract type:
  - Month-to-month: 6 months
  - One year: 12 months
  - Two year: 24 months
- **CLV = MonthlyCharges × ExpectedTenure**

## Deployment
Deploy to Streamlit Community Cloud:
1. Push to GitHub
2. Connect at https://share.streamlit.io
3. Select repo and branch