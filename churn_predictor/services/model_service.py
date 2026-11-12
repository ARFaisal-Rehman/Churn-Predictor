from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ChurnModelService:
    def __init__(self, model_path, random_state: int = 42, test_size: float = 0.25):
        self.model_path = model_path
        self.random_state = random_state
        self.test_size = test_size

    def _feature_columns(self, frame: pd.DataFrame) -> List[str]:
        ignored = {"customer_id", "snapshot_date", "churned"}
        # Raw timestamps (e.g. first/last purchase) can't go into the model;
        # their information is already captured by recency/tenure features.
        return [
            col
            for col in frame.columns
            if col not in ignored and not pd.api.types.is_datetime64_any_dtype(frame[col])
        ]

    def _build_preprocessor(self, feature_columns: List[str]) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    feature_columns,
                )
            ]
        )

    def _candidate_models(self) -> Dict[str, Pipeline]:
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=3,
                n_estimators=250,
                random_state=self.random_state,
            ),
        }
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                
                eval_metric="logloss",
                n_jobs=-1,
            )
        return models

    def train(self, training_frame: pd.DataFrame) -> Dict[str, object]:
        if "churned" not in training_frame.columns:
            raise ValueError("Training frame must include a churned target column.")

        feature_columns = self._feature_columns(training_frame)
        X = training_frame[feature_columns]
        y = training_frame["churned"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        candidates = self._candidate_models()
        model_results: Dict[str, Dict[str, float]] = {}
        best_name = None
        best_pipeline = None
        best_auc = -np.inf

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for model_name, model in candidates.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self._build_preprocessor(feature_columns)),
                    ("model", model),
                ]
            )

            fold_scores: List[float] = []
            for train_index, valid_index in cv.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_index]
                X_fold_valid = X_train.iloc[valid_index]
                y_fold_train = y_train.iloc[train_index]
                y_fold_valid = y_train.iloc[valid_index]

                fold_pipeline = clone(pipeline)
                fold_pipeline.fit(X_fold_train, y_fold_train)
                fold_prob = fold_pipeline.predict_proba(X_fold_valid)[:, 1]
                fold_scores.append(float(roc_auc_score(y_fold_valid, fold_prob)))

            pipeline.fit(X_train, y_train)
            test_prob = pipeline.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= 0.5).astype(int)

            metrics = {
                "cv_roc_auc": round(float(np.mean(fold_scores)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
                "accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
                "precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
            }

            model_results[model_name] = metrics

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = model_name
                best_pipeline = pipeline

        if best_pipeline is None or best_name is None:
            raise RuntimeError("Could not train a usable churn model.")

        feature_importance = self._feature_importance(best_pipeline, X_test, y_test, feature_columns)

        bundle = {
            "best_model_name": best_name,
            "pipeline": best_pipeline,
            "feature_columns": feature_columns,
            "model_results": model_results,
            "feature_importance": feature_importance,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)
        return bundle

    def load(self) -> Dict[str, object]:
        return joblib.load(self.model_path)

    def predict(self, bundle: Dict[str, object], scoring_frame: pd.DataFrame) -> pd.DataFrame:
        feature_columns = bundle["feature_columns"]
        pipeline = bundle["pipeline"]

        scored = scoring_frame.copy()
        probabilities = pipeline.predict_proba(scored[feature_columns])[:, 1]
        scored["churn_probability"] = probabilities
        scored["risk_category"] = pd.cut(
            scored["churn_probability"],
            bins=[0.0, 0.35, 0.65, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        ).astype(str)

        # Pre-calculate top_reasons so recommended_action can use it
        scored["top_reasons"] = scored.apply(self._top_reasons, axis=1)
        scored["recommended_action"] = scored.apply(self._recommended_action, axis=1)

        return scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    def segment(self, scored_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        segment_features = [
            "recency_days",
            "order_count",
            "total_revenue",
            "orders_last_90d",
            "revenue_last_90d",
            "return_rate",
        ]

        X = scored_frame[segment_features].copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=20)
        labels = kmeans.fit_predict(scaled)

        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(scaled)

        segmented = scored_frame.copy()
        segmented["segment_id"] = labels

        ranking = (
            segmented.groupby("segment_id")["churn_probability"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        segment_names = {
            ranking[0]: "Core",
            ranking[1]: "Watchlist",
            ranking[2]: "At Risk",
        }
        segmented["segment_name"] = segmented["segment_id"].map(segment_names)

        counts = (
            segmented["segment_name"]
            .value_counts()
            .reindex(["Core", "Watchlist", "At Risk"], fill_value=0)
            .to_dict()
        )
        pca_points = [
            {
                "x": round(float(point[0]), 4),
                "y": round(float(point[1]), 4),
                "segment": segmented.iloc[idx]["segment_name"],
            }
            for idx, point in enumerate(coords)
        ]

        segment_summary = (
            segmented.groupby("segment_name")
            .agg(
                customers=("customer_id", "count"),
                avg_risk=("churn_probability", "mean"),
                avg_revenue=("total_revenue", "mean"),
                avg_orders=("order_count", "mean"),
            )
            .round(2)
            .reset_index()
            .to_dict("records")
        )

        return segmented, {
            "counts": counts,
            "pca_points": pca_points[:500],
            "summary": segment_summary,
        }

    def _feature_importance(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_columns: List[str],
    ) -> List[Dict[str, float]]:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=self.random_state,
            scoring="roc_auc",
        )
        ranked = sorted(
            zip(feature_columns, result.importances_mean),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return [
            {"feature": feature, "importance": round(float(score), 4)}
            for feature, score in ranked[:12]
        ]

    def _top_reasons(self, row: pd.Series) -> List[str]:
        reason_rules = [
            ("Long gap since last order", row.get("recency_days", 0)),
            ("Weak recent order activity", -row.get("orders_last_30d", 0)),
            ("Weak recent revenue trend", -row.get("revenue_last_30d", 0)),
            ("High return behavior", row.get("return_rate", 0) * 100),
            ("Low order frequency", -row.get("order_frequency_30d", 0)),
            ("Low product engagement", -row.get("products_last_90d", 0)),
        ]
        ordered = sorted(reason_rules, key=lambda item: item[1], reverse=True)
        return [label for label, _ in ordered[:3]]

    def _recommended_action(self, row: pd.Series) -> str:
        reasons = row.get("top_reasons", [])
        risk = row.get("risk_category", "Low")

        if risk == "High":
            if "Long gap since last order" in reasons:
                return "Launch an urgent win-back sequence with a time-boxed incentive."
            if "High return behavior" in reasons:
                return "Pair outreach with service recovery and product-fit review."
            return "Assign priority retention outreach and a personalized offer."

        if risk == "Medium":
            if "Weak recent revenue trend" in reasons:
                return "Use a targeted cross-sell bundle to raise engagement."
            return "Run a nurture campaign with category-specific recommendations."

        return "Keep this customer in a low-cost loyalty journey and monitor trend changes."
