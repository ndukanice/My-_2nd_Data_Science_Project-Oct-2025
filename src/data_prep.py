from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv(RAW_DIR / "Telco-Customer-Churn.csv")
df = df.drop(['Unnamed: 0', 'customerID'], axis=1, errors='ignore')

# Handle TotalCharges missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
missing_tc = df["TotalCharges"].isna()
df.loc[missing_tc, "TotalCharges"] = (df.loc[missing_tc, "MonthlyCharges"] * df.loc[missing_tc, "tenure"]).round(2)

# Feature Engineering
# 1. tenure_bucket
df["tenure_bucket"] = pd.cut(
    df["tenure"], 
    bins=[-1, 6, 12, 24, np.inf], 
    labels=["0-6m", "6-12m", "12-24m", "24m+"]
)

# 2. services_count
services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
temp_services = df[services].replace({"No internet service": "No", "No phone service": "No"})
df["services_count"] = (temp_services == "Yes").sum(axis=1).astype(int)

# 3. monthly_to_total_ratio
denom = np.maximum(1, df["tenure"] * df["MonthlyCharges"])
df["monthly_to_total_ratio"] = (df["TotalCharges"] / denom).clip(0, 10)

# 4. internet_but_no_tech_support
df["internet_but_no_tech_support"] = (
    (df["InternetService"] != "No") & (df["TechSupport"] == "No")
).astype(int)

# 5. ExpectedTenure (based on Contract)
contract_to_months = {"Month-to-month": 6, "One year": 12, "Two year": 24}
df["ExpectedTenure"] = df["Contract"].map(contract_to_months).fillna(6).astype(int)

# 6. CLV
df["CLV"] = (df["MonthlyCharges"] * df["ExpectedTenure"]).round(2)

# Convert Churn to 0/1
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Save full processed data (for CLV analysis)
df.to_csv(PROC_DIR / "processed_full.csv", index=False)

# Split features and target
feature_cols = [c for c in df.columns if c not in ["Churn", "CLV", "ExpectedTenure"]]
X = df[feature_cols].copy()
y = df["Churn"].copy()

# 60/20/20 stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Save splits
X_train.to_csv(PROC_DIR / "X_train.csv", index=False)
X_val.to_csv(PROC_DIR / "X_val.csv", index=False)
X_test.to_csv(PROC_DIR / "X_test.csv", index=False)
y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
y_val.to_csv(PROC_DIR / "y_val.csv", index=False)
y_test.to_csv(PROC_DIR / "y_test.csv", index=False)

print(f"✓ Data preparation complete")
print(f"✓ Files saved to: {PROC_DIR}")
print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")