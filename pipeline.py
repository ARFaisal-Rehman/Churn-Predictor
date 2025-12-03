import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, replace spaces with underscores, and strip for consistency."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Normalize common id column names
    if 'customerid' in df.columns and 'customer_id' not in df.columns:
        df['customer_id'] = df['customerid']
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle TotalCharges numeric coercion, drop obvious duplicates, trim strings."""
    df = df.copy()
    # Coerce total_charges to numeric if present
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    if 'monthly_charges' in df.columns:
        df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    # Trim string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop duplicate customer rows if any
    if 'customer_id' in df.columns:
        df = df.drop_duplicates(subset=['customer_id'])
    else:
        df = df.drop_duplicates()

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple churn indicator features if source columns exist."""
    df = df.copy()

    # Charges to tenure ratio
    if {'monthly_charges', 'tenure'}.issubset(df.columns):
        df['charges_tenure_ratio'] = df['monthly_charges'] / df['tenure'].replace(0, np.nan)
        df['charges_tenure_ratio'] = df['charges_tenure_ratio'].fillna(0.0)

    # Long-term contract indicator
    if 'contract' in df.columns:
        df['is_long_term_contract'] = df['contract'].str.lower().isin(['one_year', 'two_year', '1_year', '2_year', 'one year', 'two year']).astype(int)

    # High charges indicator
    if 'monthly_charges' in df.columns:
        df['is_high_charges'] = (df['monthly_charges'] > df['monthly_charges'].median()).astype(int)

    # Paperless billing indicator
    if 'paperless_billing' in df.columns:
        df['paperless_billing_flag'] = df['paperless_billing'].str.lower().isin(['yes', 'true']).astype(int)

    return df


def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """Build a sklearn preprocessing pipeline based on dataframe schema."""
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude obvious identifier columns from preprocessing
    exclude_cols = {'customer_id'}
    numeric_features = [c for c in numeric_features if c not in exclude_cols]
    categorical_features = [c for c in categorical_features if c not in exclude_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, Any]:
    """Apply preprocessing to features, returning transformed array and fitted preprocessor."""
    preprocessor, _, _ = build_preprocessor(df)
    X_transformed = preprocessor.fit_transform(df)
    return X_transformed, preprocessor


def train_test_split_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) == 2 else None)
    return X_train, X_test, y_train, y_test


def run_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """Run end-to-end data pipeline and return artifacts for downstream analysis."""
    df_std = standardize_columns(df)
    df_clean = clean_data(df_std)
    df_feat = feature_engineer(df_clean)

    # Identify target column if available
    target_candidates = [
        'churn', 'is_churn', 'churn_flag'
    ]
    target_col = next((c for c in target_candidates if c in df_feat.columns), None)

    artifacts: Dict[str, Any] = {
        'df_processed': df_feat,
        'target_col': target_col
    }

    if target_col:
        X = df_feat.drop(columns=[target_col])
        X_transformed, preprocessor = preprocess_features(X)
        artifacts.update({
            'X_transformed': X_transformed,
            'preprocessor': preprocessor
        })

    return artifacts

