"""Model selection, calibration, out-of-time evaluation, explanations, segments.

Protocol (all snapshots come from TransactionDataService):
  train snapshots  -> 5-fold StratifiedGroupKFold CV (groups = customer) picks the model
  calibration snap -> isotonic calibration of the chosen model (a later period)
  test snapshot    -> out-of-time evaluation, touched once
  scoring snapshot -> today's customers, scored with the calibrated model
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

NON_FEATURES = {"customer_id", "snapshot_date", "churned"}

# Plain-language names for explanations and the UI.
FEATURE_LABELS = {
    "recency_days": "Days since last order",
    "recency_vs_cadence": "Overdue vs own buying rhythm",
    "orders_last_30d": "Orders in last 30 days",
    "orders_last_60d": "Orders in last 60 days",
    "orders_last_90d": "Orders in last 90 days",
    "orders_last_180d": "Orders in last 180 days",
    "revenue_last_30d": "Spend in last 30 days",
    "revenue_last_60d": "Spend in last 60 days",
    "revenue_last_90d": "Spend in last 90 days",
    "revenue_last_180d": "Spend in last 180 days",
    "products_last_30d": "Products bought in last 30 days",
    "products_last_60d": "Products bought in last 60 days",
    "products_last_90d": "Products bought in last 90 days",
    "products_last_180d": "Products bought in last 180 days",
    "order_count": "Orders in the past year",
    "orders_per_month": "Orders per month",
    "tenure_days": "Days since first order",
    "total_revenue": "Spend in the past year",
    "revenue_per_day": "Average spend per day",
    "value_per_window": "Typical spend per 60 days",
    "avg_order_value": "Average order value",
    "median_order_value": "Median order value",
    "max_order_value": "Largest order",
    "avg_items_per_order": "Items per order",
    "total_items": "Items bought",
    "avg_distinct_products": "Distinct products per order",
    "unique_products": "Distinct products bought",
    "avg_unit_price": "Average unit price",
    "countries_served": "Countries shipped to",
    "return_count": "Returns",
    "return_value": "Value returned",
    "returned_items": "Items returned",
    "return_rate": "Returns per order",
    "return_value_ratio": "Share of spend returned",
    "avg_days_between_orders": "Average days between purchase days",
    "purchase_days": "Days with a purchase",
    "std_days_between_orders": "Irregularity of ordering",
    "revenue_trend_90_180": "Spend trend (last 90d vs prior 90d)",
}
MONEY = {"revenue_last_30d", "revenue_last_60d", "revenue_last_90d", "revenue_last_180d", "total_revenue",
         "revenue_per_day", "value_per_window", "avg_order_value", "median_order_value", "max_order_value",
         "avg_unit_price", "return_value"}


SEASON_FEATURES = {"season_sin", "season_cos"}


def feature_columns(frame: pd.DataFrame, use_season: bool = True) -> List[str]:
    skip = NON_FEATURES | (set() if use_season else SEASON_FEATURES)
    return [c for c in frame.columns if c not in skip and pd.api.types.is_numeric_dtype(frame[c])]


def format_value(feature: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if feature in MONEY:
        return f"£{value:,.0f}"
    if feature in {"return_rate", "return_value_ratio"}:
        return f"{value:.0%}"
    if feature in {"recency_vs_cadence", "revenue_trend_90_180", "orders_per_month", "avg_items_per_order",
                   "avg_distinct_products"}:
        return f"{value:.2f}"
    return f"{value:,.0f}"


class ChurnModelService:
    def __init__(self, model_path, random_state: int = 42, window_days: int = 60, use_season: bool = False):
        self.model_path = model_path
        self.random_state = random_state
        self.window_days = window_days
        self.use_season = use_season

    # -------------------------------------------------------------- models
    def candidates(self) -> Dict[str, object]:
        rs = self.random_state
        models = {
            "logistic_regression": LogisticRegression(max_iter=3000, C=0.5, random_state=rs),
            "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=20,
                                                    n_jobs=-1, random_state=rs),
            "hist_gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                     max_leaf_nodes=31, l2_regularization=1.0,
                                                                     random_state=rs),
        }
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                              colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                                              n_jobs=-1, random_state=rs)
        return models

    @staticmethod
    def pipeline(model) -> Pipeline:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if isinstance(model, LogisticRegression):
            steps.append(("scale", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    @staticmethod
    def _prep(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        # Heavy-tailed money/count features are log-scaled; trees don't care, the linear model does.
        X = frame[cols].astype(float).replace([np.inf, -np.inf], np.nan)
        return np.sign(X) * np.log1p(X.abs())

    def forward_folds(self, train: pd.DataFrame, n_folds: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Forward-chaining CV over snapshot dates: validate on each of the last `n_folds` training snapshots
        using only snapshots whose label window had closed before it. Random or customer-grouped folds mix
        periods and reward memorising seasons the model has already seen."""
        dates = pd.to_datetime(train["snapshot_date"])
        unique = sorted(dates.unique())
        folds = []
        for val_date in unique[-n_folds:]:
            tr = np.where(dates <= val_date - pd.Timedelta(days=self.window_days))[0]
            va = np.where(dates == val_date)[0]
            if len(tr) and len(va):
                folds.append((tr, va))
        return folds

    def train(self, train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> Dict[str, object]:
        cols = feature_columns(train, self.use_season)
        X_tr, y_tr, groups = self._prep(train, cols), train["churned"].astype(int), train["customer_id"]
        X_cal, y_cal = self._prep(calibration, cols), calibration["churned"].astype(int)
        X_te, y_te = self._prep(test, cols), test["churned"].astype(int)

        grouped = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                       .split(X_tr, y_tr, groups))
        forward = self.forward_folds(train)
        results: Dict[str, Dict[str, float]] = {}
        fitted: Dict[str, Pipeline] = {}
        for name, model in self.candidates().items():
            def cv(folds):
                s = []
                for tr_idx, va_idx in folds:
                    p = clone(self.pipeline(model)).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
                    s.append(roc_auc_score(y_tr.iloc[va_idx], p.predict_proba(X_tr.iloc[va_idx])[:, 1]))
                return s
            fwd, grp = cv(forward), cv(grouped)
            pipe = self.pipeline(model).fit(X_tr, y_tr)
            fitted[name] = pipe
            results[name] = {
                "forward_cv_roc_auc": round(float(np.mean(fwd)), 4),     # used for selection
                "forward_cv_roc_auc_std": round(float(np.std(fwd)), 4),
                "grouped_cv_roc_auc": round(float(np.mean(grp)), 4),     # shown for comparison only
                # Reported for transparency only; never used to choose the model.
                "oot_roc_auc": round(float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])), 4),
            }

        best_name = max(results, key=lambda n: results[n]["forward_cv_roc_auc"])
        best = fitted[best_name]
        calibration_method, calibration_cv = self._choose_calibration(best, X_cal, y_cal)
        calibrated = CalibratedClassifierCV(FrozenEstimator(best), method=calibration_method).fit(X_cal, y_cal)

        raw_te = best.predict_proba(X_te)[:, 1]
        cal_te = calibrated.predict_proba(X_te)[:, 1]
        evaluation = self._evaluate(y_te, raw_te, cal_te)
        evaluation["random_split_roc_auc"] = self._random_split_auc(clone(best), X_te, y_te)

        rng = np.random.RandomState(self.random_state)
        sample = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
        imp = permutation_importance(best, X_te.iloc[sample], y_te.iloc[sample], n_repeats=5,
                                     random_state=self.random_state, scoring="roc_auc", n_jobs=1)
        ranked = sorted(zip(cols, imp.importances_mean), key=lambda kv: kv[1], reverse=True)
        importance = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(float(v), 4)}
                      for f, v in ranked[:12]]

        bundle = {
            "best_model_name": best_name,
            "model": best,
            "calibrated": calibrated,
            "feature_columns": cols,
            "medians": X_tr.median().to_dict(),
            "raw_medians": train[cols].median().to_dict(),
            "model_results": results,
            "selection": "forward-chaining CV over snapshot dates",
            "calibration_method": calibration_method,
            "calibration_cv_log_loss": calibration_cv,
            "use_season": self.use_season,
            "evaluation": evaluation,
            "feature_importance": importance,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)
        return bundle

    def _choose_calibration(self, model, X_cal, y_cal) -> Tuple[str, Dict[str, float]]:
        """Pick sigmoid or isotonic by 5-fold log loss *inside* the calibration snapshot. Isotonic can
        overfit small sets and emit hard 0/1 probabilities; sigmoid is smoother. The test set is not used."""
        from sklearn.model_selection import StratifiedKFold
        scores = {}
        for method in ("sigmoid", "isotonic"):
            losses = []
            for a, b in StratifiedKFold(5, shuffle=True, random_state=self.random_state).split(X_cal, y_cal):
                c = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal.iloc[a], y_cal.iloc[a])
                losses.append(log_loss(y_cal.iloc[b], np.clip(c.predict_proba(X_cal.iloc[b])[:, 1], 1e-6, 1 - 1e-6)))
            scores[method] = round(float(np.mean(losses)), 4)
        return min(scores, key=scores.get), scores

    def _random_split_auc(self, model, X, y) -> float:
        """The old protocol: random 75/25 split inside a single snapshot. Kept as a comparison."""
        X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=0.25, stratify=y, random_state=self.random_state)
        model.fit(X_a, y_a)
        return round(float(roc_auc_score(y_b, model.predict_proba(X_b)[:, 1])), 4)

    @staticmethod
    def _evaluate(y, raw, cal) -> Dict[str, object]:
        order = np.argsort(-cal)
        top = order[: max(1, len(order) // 10)]
        base = float(np.mean(y))
        frac_raw, mean_raw = calibration_curve(y, raw, n_bins=10, strategy="quantile")
        frac_cal, mean_cal = calibration_curve(y, cal, n_bins=10, strategy="quantile")
        return {
            "n": int(len(y)),
            "base_rate": round(base, 4),
            "roc_auc": round(float(roc_auc_score(y, cal)), 4),
            "roc_auc_uncalibrated": round(float(roc_auc_score(y, raw)), 4),
            "pr_auc": round(float(average_precision_score(y, cal)), 4),
            "brier_uncalibrated": round(float(brier_score_loss(y, raw)), 4),
            "brier_calibrated": round(float(brier_score_loss(y, cal)), 4),
            "brier_base_rate": round(float(brier_score_loss(y, np.full_like(cal, base))), 4),
            "log_loss_uncalibrated": round(float(log_loss(y, np.clip(raw, 1e-6, 1 - 1e-6))), 4),
            "log_loss_calibrated": round(float(log_loss(y, np.clip(cal, 1e-6, 1 - 1e-6))), 4),
            "precision_top_decile": round(float(np.mean(np.asarray(y)[top])), 4),
            "lift_top_decile": round(float(np.mean(np.asarray(y)[top]) / base), 2) if base else None,
            "reliability": {
                "uncalibrated": {"predicted": mean_raw.round(4).tolist(), "observed": frac_raw.round(4).tolist()},
                "calibrated": {"predicted": mean_cal.round(4).tolist(), "observed": frac_cal.round(4).tolist()},
            },
        }

    def load(self) -> Dict[str, object]:
        return joblib.load(self.model_path)

    # ------------------------------------------------------------- scoring
    def predict(self, bundle: Dict[str, object], scoring: pd.DataFrame) -> pd.DataFrame:
        cols = bundle["feature_columns"]
        X = self._prep(scoring, cols)
        scored = scoring.copy()
        scored["churn_probability"] = bundle["calibrated"].predict_proba(X)[:, 1]
        scored["risk_category"] = pd.cut(scored["churn_probability"], bins=[-0.01, 0.35, 0.65, 1.0],
                                         labels=["Low", "Medium", "High"]).astype(str)
        scored["top_reasons"] = self.explain(bundle, scoring, X)
        return scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    def explain(self, bundle, scoring: pd.DataFrame, X: pd.DataFrame, top_k: int = 3) -> List[List[Dict]]:
        """One-at-a-time occlusion: how much does each feature push this customer's risk above what it
        would be if that feature were typical (the training median)? Correlated features share credit,
        so read the reasons as 'signals', not causes."""
        model, cols, med = bundle["model"], bundle["feature_columns"], bundle["medians"]
        base = model.predict_proba(X)[:, 1]
        deltas = np.zeros((len(X), len(cols)))
        for j, col in enumerate(cols):
            X_typical = X.copy()
            X_typical[col] = med[col]
            deltas[:, j] = base - model.predict_proba(X_typical)[:, 1]
        raw_med = bundle["raw_medians"]
        out = []
        for i in range(len(X)):
            idx = [j for j in np.argsort(-deltas[i]) if deltas[i, j] > 0.005][:top_k]
            out.append([{
                "feature": cols[j],
                "label": FEATURE_LABELS.get(cols[j], cols[j]),
                "value": format_value(cols[j], scoring.iloc[i][cols[j]]),
                "typical": format_value(cols[j], raw_med[cols[j]]),
                "impact": round(float(deltas[i, j]) * 100, 1),
            } for j in idx])
        return out

    # ------------------------------------------------------------ segments
    SEGMENT_FEATURES = ["recency_days", "order_count", "total_revenue", "avg_order_value"]

    def segment(self, scored: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        X = np.log1p(scored[self.SEGMENT_FEATURES].clip(lower=0).astype(float))
        Z = StandardScaler().fit_transform(X)
        rng = np.random.RandomState(self.random_state)
        sample = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)

        silhouettes = {}
        fits = {}
        for k in range(3, 7):
            km = KMeans(n_clusters=k, n_init=20, random_state=self.random_state).fit(Z)
            silhouettes[k] = round(float(silhouette_score(Z[sample], km.labels_[sample])), 4)
            fits[k] = km
        best_k = max(silhouettes, key=silhouettes.get)
        labels = fits[best_k].labels_

        seg = scored.copy()
        seg["segment_id"] = labels
        names = self._name_segments(seg)
        seg["segment_name"] = seg["segment_id"].map(names)

        summary = (seg.groupby("segment_name")
                   .agg(customers=("customer_id", "count"), avg_risk=("churn_probability", "mean"),
                        median_recency=("recency_days", "median"), median_orders=("order_count", "median"),
                        median_spend=("total_revenue", "median"), spend_total=("total_revenue", "sum"))
                   .sort_values("avg_risk").round(3).reset_index().to_dict("records"))
        return seg, {"k": best_k, "silhouette": silhouettes, "summary": summary,
                     "counts": seg["segment_name"].value_counts().to_dict()}

    @staticmethod
    def _name_segments(seg: pd.DataFrame) -> Dict[int, str]:
        """Name clusters from their medians: recent vs lapsing, high vs low value."""
        prof = seg.groupby("segment_id").agg(r=("recency_days", "median"), f=("order_count", "median"),
                                             m=("total_revenue", "median"))
        r_mid, value = prof["r"].median(), (prof["f"].rank() + prof["m"].rank())
        v_mid = value.median()
        names, used = {}, {}
        for sid, row in prof.iterrows():
            recent = row["r"] <= r_mid
            high = value[sid] > v_mid
            base = ("Loyal high-value" if recent and high else
                    "Recent, light buyers" if recent else
                    "Slipping high-value" if high else
                    "Dormant, low-value")
            used[base] = used.get(base, 0) + 1
            names[sid] = base if used[base] == 1 else f"{base} ({used[base]})"
        return names

    # ------------------------------------------------------------- actions
    @staticmethod
    def recommended_action(row: pd.Series) -> str:
        """Simple playbook rules, stated as rules rather than model output."""
        risk, seg = row["risk_category"], str(row.get("segment_name", ""))
        reasons = {r["feature"] for r in row.get("top_reasons", [])}
        if risk == "High":
            if seg.startswith(("Loyal high-value", "Slipping high-value")):
                return "Personal call from an account manager this week; protect the relationship before discounting."
            if reasons & {"return_rate", "return_value_ratio", "return_count", "return_value"}:
                return "Service recovery first: review recent returns and product fit, then follow up."
            return "Time-boxed win-back email with a modest incentive; stop if no response in two touches."
        if risk == "Medium":
            if reasons & {"revenue_last_90d", "revenue_trend_90_180", "products_last_90d"}:
                return "Category-relevant cross-sell bundle to lift recent spend."
            return "Add to a nurture sequence with new-range and reorder reminders."
        return "No spend needed: keep in the standard newsletter and monitor."
