from pathlib import Path
import json
import pandas as pd
import numpy as np
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

CONTRACT_TENURE = {"Month-to-month": 6, "One year": 12, "Two year": 24}

def load_model(name: str = "xgb"):
    """Load a trained model"""
    return load(MODELS_DIR / f"{name}.pkl")

def load_feature_columns():
    """Load the feature columns used during training"""
    with open(MODELS_DIR / "columns.json", "r") as f:
        return json.load(f)["feature_columns"]

def engineer_row(row: pd.Series) -> pd.Series:
    """Apply feature engineering to a single row (matches data_prep.py logic)"""
    # tenure_bucket
    t = float(row["tenure"])
    if t <= 6: 
        row["tenure_bucket"] = "0-6m"
    elif t <= 12: 
        row["tenure_bucket"] = "6-12m"
    elif t <= 24: 
        row["tenure_bucket"] = "12-24m"
    else: 
        row["tenure_bucket"] = "24m+"

    # services_count
    svc = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
           "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    temp = row[svc].replace({"No internet service": "No", "No phone service": "No"})
    row["services_count"] = int((temp == "Yes").sum())

    # monthly_to_total_ratio
    denom = max(1.0, float(row["tenure"]) * float(row["MonthlyCharges"]))
    row["monthly_to_total_ratio"] = float(row["TotalCharges"]) / denom

    # internet_but_no_tech_support flag
    row["internet_but_no_tech_support"] = int(
        row["InternetService"] != "No" and row["TechSupport"] == "No"
    )

    # ExpectedTenure + CLV
    expected = CONTRACT_TENURE.get(row["Contract"], 6)
    row["ExpectedTenure"] = expected
    row["CLV"] = round(float(row["MonthlyCharges"]) * expected, 2)
    
    return row

def predict_single(form_dict: dict, model_name: str = "xgb"):
    """
    Predict churn for a single customer.
    
    Args:
        form_dict: Dictionary with raw customer data
        model_name: Model to use (logistic, rf, or xgb)
    
    Returns:
        tuple: (churn_probability, CLV)
    """
    # Load feature columns from training
    cols = load_feature_columns()
    
    # Convert to Series and apply feature engineering
    row = pd.Series(form_dict)
    row = engineer_row(row)
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([row]).reindex(columns=cols, fill_value=0)
    
    # Load model and predict
    model = load_model(model_name)
    proba = float(model.predict_proba(X)[:, 1][0])
    
    return proba, float(row["CLV"])

if __name__ == "__main__":
    # Test with a high-risk customer
    high_risk_customer = {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 105.0,
        "TotalCharges": 105.0,
    }
    
    proba, clv = predict_single(high_risk_customer, model_name="xgb")
    print(f"Churn Probability: {proba*100:.1f}%")
    print(f"CLV: ${clv:,.2f}")
    print(f"Risk Level: {'High' if proba > 0.66 else ('Medium' if proba > 0.33 else 'Low')}")