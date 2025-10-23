from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

def get_feature_names_from_preprocessor(pre, input_cols: List[str]) -> List[str]:
    """Extract feature names after preprocessing (OneHot expansion)"""
    try:
        names = pre.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback: manually reconstruct feature names
        cat_cols = pre.transformers_[1][2]
        num_cols = pre.transformers_[0][2]
        ohe = pre.transformers_[1][1]
        cat_expanded = []
        for base, cats in zip(cat_cols, ohe.categories_):
            cat_expanded += [f"{base}_{c}" for c in cats]
        return list(num_cols) + cat_expanded

def tree_feature_importance(pipeline, input_cols: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models (RF, XGBoost).
    Groups one-hot encoded features back to base feature names.
    """
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    
    # Get feature names after preprocessing
    names = get_feature_names_from_preprocessor(pre, input_cols)
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    
    # Create DataFrame with encoded feature names
    df = pd.DataFrame({"feature_enc": names, "importance": importances})
    
    # Extract base feature name (before the underscore from one-hot encoding)
    df["base"] = df["feature_enc"].str.split("_", n=1).str[0]
    
    # Group by base feature and sum importances
    agg = df.groupby("base")["importance"].sum().sort_values(ascending=False).reset_index()
    agg.rename(columns={"base": "feature"}, inplace=True)
    
    return agg

def logistic_importance(pipeline, input_cols: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from Logistic Regression using absolute coefficients.
    Groups one-hot encoded features back to base feature names.
    """
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    
    # Get feature names after preprocessing
    names = get_feature_names_from_preprocessor(pre, input_cols)
    
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["feature", "importance"])
    
    # Use absolute values of coefficients as importance
    coefs = np.abs(clf.coef_.ravel())
    
    # Create DataFrame with encoded feature names
    df = pd.DataFrame({"feature_enc": names, "coef_abs": coefs})
    
    # Extract base feature name (before the underscore from one-hot encoding)
    df["base"] = df["feature_enc"].str.split("_", n=1).str[0]
    
    # Group by base feature and sum absolute coefficients
    agg = df.groupby("base")["coef_abs"].sum().sort_values(ascending=False).reset_index()
    agg.rename(columns={"base": "feature", "coef_abs": "importance"}, inplace=True)
    
    return agg