import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, Any, List, Tuple


def _binarize(df: pd.DataFrame, max_unique: int = 30) -> pd.DataFrame:
    """Convert categorical columns to one-hot boolean indicators. Limit high-card columns."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Limit columns with too many unique values
    cat_cols = [c for c in cat_cols if df[c].nunique() <= max_unique]
    if not cat_cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cat_cols].astype(str), drop_first=False).astype(bool)


def mine_associations(df: pd.DataFrame, target_col: str = 'churn', min_support: float = 0.05, min_confidence: float = 0.3) -> Dict[str, Any]:
    """Simple association mining: compute pairwise rules X -> churn using boolean one-hot inputs.
    If target_col is not present, infer using 'churn_probability' threshold.
    """
    df = df.copy()

    # Target definition
    if target_col not in df.columns:
        if 'churn_probability' in df.columns:
            target = (df['churn_probability'] >= 0.5).astype(int)
        else:
            # No target available
            return {'rules': [], 'info': 'No churn/target column present'}
    else:
        target = df[target_col].astype(str).str.lower().isin(['yes', 'true', '1']).astype(int)

    X_bin = _binarize(df)
    if X_bin.empty:
        return {'rules': [], 'info': 'No suitable categorical columns for association mining'}

    n = len(df)
    support_y = target.mean()

    # Compute supports for individual items
    item_support = X_bin.sum(axis=0) / n
    # Keep frequent items
    frequent_items = item_support[item_support >= min_support].index.tolist()
    X_freq = X_bin[frequent_items]

    rules: List[Dict[str, Any]] = []
    for item in frequent_items:
        x = X_freq[item]
        joint_support = ((x) & (target == 1)).sum() / n
        conf = joint_support / item_support[item] if item_support[item] > 0 else 0
        lift = conf / support_y if support_y > 0 else np.nan
        if conf >= min_confidence:
            rules.append({
                'antecedent': item,
                'consequent': 'churn',
                'support': round(joint_support, 4),
                'confidence': round(conf, 4),
                'lift': round(lift, 4)
            })

    # Sort by lift then confidence
    rules = sorted(rules, key=lambda r: (r['lift'], r['confidence']), reverse=True)
    return {
        'rules': rules[:50],
        'support_y': round(float(support_y), 4),
        'total_items': len(frequent_items)
    }


def run_association_analysis(df: pd.DataFrame, target_col: str = 'churn') -> Dict[str, Any]:
    return mine_associations(df, target_col=target_col)

