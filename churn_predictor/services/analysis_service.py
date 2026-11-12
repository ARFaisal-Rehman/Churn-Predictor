from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data_service import TransactionDataService
from .model_service import ChurnModelService


def _to_serializable(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Value of type {type(value)!r} is not JSON serializable")


class AnalysisService:
    def __init__(self, config):
        self.config = config
        self.data_service = TransactionDataService(
            config["DATA_PATH"],
            prediction_window_days=config["PREDICTION_WINDOW_DAYS"],
        )
        self.model_service = ChurnModelService(
            config["MODEL_PATH"],
            random_state=config["RANDOM_STATE"],
            test_size=config["TEST_SIZE"],
        )

    def get_or_create_analysis(self) -> Dict[str, Any]:
        if self.config["ANALYSIS_PATH"].exists() and self.config["PREDICTIONS_PATH"].exists():
            return self.load_analysis()
        return self.refresh_analysis(force_retrain=True)

    def refresh_analysis(self, force_retrain: bool = False) -> Dict[str, Any]:
        prepared = self.data_service.prepare_datasets()

        if force_retrain or not self.config["MODEL_PATH"].exists():
            model_bundle = self.model_service.train(prepared.training_frame)
        else:
            model_bundle = self.model_service.load()

        scored = self.model_service.predict(model_bundle, prepared.scoring_frame)
        scored, segmentation = self.model_service.segment(scored)

        analysis = self._build_analysis(
            prepared=prepared,
            scored=scored,
            model_bundle=model_bundle,
            segmentation=segmentation,
        )

        self.config["ARTIFACTS_DIR"].mkdir(parents=True, exist_ok=True)
        prepared.training_frame.to_csv(self.config["TRAINING_PATH"], index=False)
        scored.to_csv(self.config["PREDICTIONS_PATH"], index=False)
        with open(self.config["ANALYSIS_PATH"], "w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, default=_to_serializable)

        return analysis

    def load_analysis(self) -> Dict[str, Any]:
        with open(self.config["ANALYSIS_PATH"], "r", encoding="utf-8") as handle:
            analysis = json.load(handle)

        predictions = pd.read_csv(self.config["PREDICTIONS_PATH"])
        predictions["top_reasons"] = predictions["top_reasons"].apply(self._safe_list)
        analysis["customers"] = predictions.to_dict("records")
        return analysis

    def get_customer(self, customer_id: str) -> Dict[str, Any] | None:
        analysis = self.get_or_create_analysis()
        return next((item for item in analysis["customers"] if str(item["customer_id"]) == str(customer_id)), None)

    def _build_analysis(self, prepared, scored: pd.DataFrame, model_bundle: Dict[str, Any], segmentation: Dict[str, Any]):
        total_customers = int(len(scored))
        risk_counts = scored["risk_category"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)

        top_reasons = (
            pd.Series([reason for reasons in scored["top_reasons"] for reason in reasons])
            .value_counts()
            .head(6)
        )

        top_customers = scored[
            [
                "customer_id",
                "churn_probability",
                "risk_category",
                "segment_name",
                "total_revenue",
                "order_count",
                "recency_days",
                "recommended_action",
            ]
        ].head(15)

        risk_bands = {
            "labels": ["High", "Medium", "Low"],
            "values": [int(risk_counts["High"]), int(risk_counts["Medium"]), int(risk_counts["Low"])],
        }

        feature_importance = model_bundle["feature_importance"]
        revenue_by_risk = (
            scored.groupby("risk_category")["total_revenue"]
            .sum()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .round(2)
            .to_dict()
        )

        trend_frame = (
            scored.groupby(pd.cut(scored["recency_days"], bins=[-1, 30, 60, 90, 180, 365, 2000]))
            .agg(avg_risk=("churn_probability", "mean"), customers=("customer_id", "count"))
            .reset_index()
        )

        customers = scored.copy()
        customers["top_reasons"] = customers["top_reasons"].apply(list)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "dataset": {
                "path": str(self.config["DATA_PATH"]),
                "reference_date": prepared.reference_date.strftime("%Y-%m-%d"),
                "observation_date": prepared.observation_date.strftime("%Y-%m-%d"),
                "prediction_window_days": self.config["PREDICTION_WINDOW_DAYS"],
                "overview": self.data_service.build_overview(prepared.transactions),
            },
            "summary": {
                "total_customers": total_customers,
                "average_risk": round(float(scored["churn_probability"].mean()) * 100, 2),
                "high_risk": int(risk_counts["High"]),
                "medium_risk": int(risk_counts["Medium"]),
                "low_risk": int(risk_counts["Low"]),
                "estimated_revenue_at_risk": round(
                    float(scored.loc[scored["risk_category"] == "High", "total_revenue"].sum()),
                    2,
                ),
            },
            "model": {
                "best_model": model_bundle["best_model_name"],
                "metrics": model_bundle["model_results"],
                "feature_importance": feature_importance,
                "training_rows": int(len(prepared.training_frame)),
                "churn_rate": round(float(prepared.training_frame["churned"].mean()) * 100, 2),
            },
            "charts": {
                "risk_bands": risk_bands,
                "top_reasons": {
                    "labels": top_reasons.index.tolist(),
                    "values": [int(value) for value in top_reasons.tolist()],
                },
                "revenue_by_risk": {
                    "labels": ["High", "Medium", "Low"],
                    "values": [float(revenue_by_risk["High"]), float(revenue_by_risk["Medium"]), float(revenue_by_risk["Low"])],
                },
                "feature_importance": {
                    "labels": [item["feature"] for item in feature_importance],
                    "values": [item["importance"] for item in feature_importance],
                },
                "segment_distribution": {
                    "labels": list(segmentation["counts"].keys()),
                    "values": list(segmentation["counts"].values()),
                },
                "segment_pca": segmentation["pca_points"],
                "recency_vs_risk": {
                    "labels": [str(item) for item in trend_frame["recency_days"].tolist()],
                    "values": [round(float(value) * 100, 2) for value in trend_frame["avg_risk"].tolist()],
                    "customers": [int(value) for value in trend_frame["customers"].tolist()],
                },
            },
            "segments": segmentation["summary"],
            "top_customers": top_customers.to_dict("records"),
            "customers": customers.to_dict("records"),
        }

    def _safe_list(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value.replace("'", '"'))
            except Exception:
                cleaned = value.strip("[]")
                if not cleaned:
                    return []
                return [item.strip().strip("'").strip('"') for item in cleaned.split(",")]
        return []
