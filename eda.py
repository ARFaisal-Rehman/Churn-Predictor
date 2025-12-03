import pandas as pd
import numpy as np
from typing import Dict, Any


def statistical_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    numeric = df.select_dtypes(include=['number'])
    categorical = df.select_dtypes(include=['object', 'category'])

    summary['numeric_describe'] = numeric.describe().to_dict()
    summary['categorical_top'] = {col: df[col].value_counts().head(10).to_dict() for col in categorical.columns}
    return summary


def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=['number'])
    if numeric.empty:
        return {'correlations': {}}
    corr = numeric.corr()
    # Flatten and keep top pairs
    corr_pairs = []
    for i, col_i in enumerate(corr.columns):
        for j, col_j in enumerate(corr.columns):
            if j <= i:
                continue
            corr_pairs.append({'pair': (col_i, col_j), 'value': corr.iloc[i, j]})
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['value']), reverse=True)[:20]
    return {
        'correlations': corr.to_dict(),
        'top_pairs': corr_pairs
    }


def churn_trends(df: pd.DataFrame) -> Dict[str, Any]:
    trends = {}
    # If we have tenure or a datetime column to simulate time-series
    if 'tenure' in df.columns and 'churn_probability' in df.columns:
        # Group by tenure buckets
        bins = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, np.inf], labels=['0-6', '6-12', '12-24', '24-48', '48+'])
        grp = df.groupby(bins)['churn_probability'].mean().round(4)
        trends['churn_by_tenure'] = grp.to_dict()

    # If we have date (e.g., signup_date or month), compute monthly churn rate
    date_cols = [c for c in df.columns if 'date' in c or 'month' in c]
    for dc in date_cols:
        try:
            s = pd.to_datetime(df[dc], errors='coerce')
            if s.notna().sum() > 0:
                df2 = df.copy()
                df2['_dt'] = s.dt.to_period('M').astype(str)
                trends[f'churn_by_{dc}_month'] = df2.groupby('_dt')['churn_probability'].mean().round(4).to_dict()
                break
        except Exception:
            continue
    return trends


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'summaries': statistical_summaries(df),
        'correlations': correlation_analysis(df),
        'trends': churn_trends(df)
    }

