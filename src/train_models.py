from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_splits():
    """Load train/val/test splits"""
    X_train = pd.read_csv(PROC_DIR / "X_train.csv")
    X_val = pd.read_csv(PROC_DIR / "X_val.csv")
    X_test = pd.read_csv(PROC_DIR / "X_test.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(PROC_DIR / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(PROC_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for categorical and numerical features"""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def train_and_eval(clf: Pipeline, X_tr, y_tr, X_te, y_te, name: str):
    """Train model and evaluate on test set"""
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_te, preds), 4),
        "recall": round(recall_score(y_te, preds), 4),
        "f1": round(f1_score(y_te, preds), 4),
        "auc": round(roc_auc_score(y_te, proba), 4),
    }
    
    # Save model
    dump(clf, MODELS_DIR / f"{name}.pkl")
    print(f"{name}: {metrics}")
    return metrics

def main():
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    
    # Build preprocessor
    preprocessor = build_preprocessor(X_train)
    
    # Combine train+val for final training
    X_tr = pd.concat([X_train, X_val], axis=0)
    y_tr = pd.concat([y_train, y_val], axis=0)
    
    # Calculate class imbalance weight
    pos_weight = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
    
    # Define models with class imbalance handling
    models = {
        "logistic": LogisticRegression(
            solver="liblinear", 
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=float(pos_weight),
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=0
        ),
    }
    
    # Train and evaluate all models
    metrics_all = {}
    for name, base_model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("clf", base_model)])
        metrics = train_and_eval(pipe, X_tr, y_tr, X_test, y_test, name)
        metrics_all[name] = metrics
    
    # Save metrics
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    
    # Save column info for predict.py
    cols_info = {"feature_columns": X_train.columns.tolist()}
    with open(MODELS_DIR / "columns.json", "w") as f:
        json.dump(cols_info, f, indent=2)
    
    print(f"\n✓ Models saved to: {MODELS_DIR}")
    print(f"✓ Metrics saved to: {MODELS_DIR / 'metrics.json'}")
    
    # Check recall requirement
    for name, m in metrics_all.items():
        if m["recall"] < 0.6:
            print(f"⚠️  WARNING: {name} recall ({m['recall']:.1%}) is below 60% threshold")

if __name__ == "__main__":
    main()