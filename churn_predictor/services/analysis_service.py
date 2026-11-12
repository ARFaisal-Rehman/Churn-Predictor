"""Runs the pipeline end to end and caches what the dashboard needs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data_service import TransactionDataService
from .model_service import ChurnModelService, FEATURE_LABELS


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Value of type {type(value)!r} is not JSON serializable")


class AnalysisService:
    CUSTOMER_FIELDS = ["customer_id", "churn_probability", "risk_category", "segment_name", "total_revenue",
                       "value_per_window", "order_count", "recency_days", "recommended_action"]

    def __init__(self, config):
        self.config = config
        self.data_service = TransactionDataService(
            config["DATA_PATH"],
            prediction_window_days=config["PREDICTION_WINDOW_DAYS"],
            active_lookback_days=config["ACTIVE_LOOKBACK_DAYS"],
            snapshot_step_days=config["SNAPSHOT_STEP_DAYS"],
        )
        self.model_service = ChurnModelService(config["MODEL_PATH"], random_state=config["RANDOM_STATE"],
                                               window_days=config["PREDICTION_WINDOW_DAYS"],
                                               use_season=config["USE_SEASON_FEATURES"])
        self._cache: Dict[str, Any] | None = None
        self._customers: Dict[str, Dict[str, Any]] | None = None

    # ------------------------------------------------------------------ API
    def get_or_create_analysis(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if self.config["ANALYSIS_PATH"].exists() and self.config["PREDICTIONS_PATH"].exists():
            return self.load_analysis()
        return self.refresh_analysis(force_retrain=True)

    def refresh_analysis(self, force_retrain: bool = False) -> Dict[str, Any]:
        prepared = self.data_service.prepare_datasets()
        if force_retrain or not self.config["MODEL_PATH"].exists():
            bundle = self.model_service.train(prepared.train, prepared.calibration, prepared.test)
        else:
            bundle = self.model_service.load()

        scored = self.model_service.predict(bundle, prepared.scoring)
        scored, segmentation = self.model_service.segment(scored)
        scored["recommended_action"] = scored.apply(self.model_service.recommended_action, axis=1)

        analysis = self._build_analysis(prepared, scored, bundle, segmentation)
        self.config["ARTIFACTS_DIR"].mkdir(parents=True, exist_ok=True)
        out = scored.copy()
        out["top_reasons"] = out["top_reasons"].apply(json.dumps)
        out.to_csv(self.config["PREDICTIONS_PATH"], index=False)
        with open(self.config["ANALYSIS_PATH"], "w", encoding="utf-8") as fh:
            json.dump(analysis, fh, indent=1, default=_json_default)
        with open(self.config["METRICS_PATH"], "w", encoding="utf-8") as fh:
            json.dump({"dataset": analysis["dataset"], "model": analysis["model"],
                       "segments": analysis["segments"]}, fh, indent=1, default=_json_default)
        self._cache, self._customers = None, None
        return self.load_analysis()

    def load_analysis(self) -> Dict[str, Any]:
        with open(self.config["ANALYSIS_PATH"], "r", encoding="utf-8") as fh:
            self._cache = json.load(fh)
        return self._cache

    def get_customer(self, customer_id: str) -> Dict[str, Any] | None:
        if self._customers is None:
            self.get_or_create_analysis()
            frame = pd.read_csv(self.config["PREDICTIONS_PATH"], dtype={"customer_id": str})
            frame["top_reasons"] = frame["top_reasons"].apply(json.loads)
            frame = frame.astype(object).where(frame.notna(), None)
            self._customers = {r["customer_id"]: r for r in frame.to_dict("records")}
        return self._customers.get(str(customer_id))

    # ------------------------------------------------------------ builders
    def _build_analysis(self, prepared, scored: pd.DataFrame, bundle, segmentation) -> Dict[str, Any]:
        plan = prepared.plan
        risk_counts = scored["risk_category"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        expected_loss = float((scored["churn_probability"] * scored["value_per_window"]).sum())

        reason_counts = pd.Series([r["label"] for rs in scored.loc[scored["risk_category"] != "Low", "top_reasons"]
                                   for r in rs]).value_counts().head(8)

        customers = scored[self.CUSTOMER_FIELDS].copy()
        customers["churn_probability"] = customers["churn_probability"].round(4)
        customers["total_revenue"] = customers["total_revenue"].round(2)
        customers["value_per_window"] = customers["value_per_window"].round(2)

        train = prepared.train
        return {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "dataset": {
                "file": self.config["DATA_PATH"].name,
                "overview": self.data_service.build_overview(prepared.transactions),
                "cleaning": prepared.cleaning,
                "prediction_window_days": self.config["PREDICTION_WINDOW_DAYS"],
                "active_lookback_days": self.config["ACTIVE_LOOKBACK_DAYS"],
                "snapshots": {
                    "train": [d.strftime("%Y-%m-%d") for d in plan.train_dates],
                    "calibration": plan.calibration_date.strftime("%Y-%m-%d"),
                    "test": plan.test_date.strftime("%Y-%m-%d"),
                    "scoring": plan.scoring_date.strftime("%Y-%m-%d"),
                },
                "rows": {"train": int(len(train)), "train_customers": int(train["customer_id"].nunique()),
                         "calibration": int(len(prepared.calibration)), "test": int(len(prepared.test)),
                         "scoring": int(len(scored))},
                "churn_rate": {"train": round(float(train["churned"].mean()), 4),
                               "calibration": round(float(prepared.calibration["churned"].mean()), 4),
                               "test": round(float(prepared.test["churned"].mean()), 4)},
            },
            "model": {
                "best_model": bundle["best_model_name"],
                "results": bundle["model_results"],
                "evaluation": bundle["evaluation"],
                "feature_importance": bundle["feature_importance"],
                "n_features": len(bundle["feature_columns"]),
                "selection": bundle.get("selection"),
                "use_season": bundle.get("use_season"),
                "calibration_method": bundle.get("calibration_method"),
                "calibration_cv_log_loss": bundle.get("calibration_cv_log_loss"),
            },
            "segments": segmentation,
            "summary": {
                "customers_scored": int(len(scored)),
                "high_risk": int(risk_counts["High"]),
                "medium_risk": int(risk_counts["Medium"]),
                "low_risk": int(risk_counts["Low"]),
                "average_risk": round(float(scored["churn_probability"].mean()), 4),
                "expected_churners": round(float(scored["churn_probability"].sum()), 1),
                "expected_revenue_loss": round(expected_loss, 2),
            },
            "charts": {
                "top_reasons": {"labels": reason_counts.index.tolist(), "values": reason_counts.astype(int).tolist()},
                "risk_histogram": np.histogram(scored["churn_probability"], bins=20, range=(0, 1))[0].tolist(),
            },
            "feature_labels": FEATURE_LABELS,
            "customers": customers.to_dict("records"),
        }
