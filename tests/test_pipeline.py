"""Pipeline tests on a small synthetic transaction log (no real data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_predictor.services.data_service import TransactionDataService


def synthetic_log(n_customers: int = 160, seed: int = 0) -> pd.DataFrame:
    """Two years of orders. Half the customers stop buying part-way through."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-01-01")
    rows = []
    for c in range(n_customers):
        stop = 730 if c % 2 else int(rng.integers(200, 700))
        gap = int(rng.integers(10, 60))
        day = int(rng.integers(0, 40))
        inv = 0
        while day < stop:
            inv += 1
            for _ in range(int(rng.integers(1, 4))):
                rows.append({"Invoice": f"{c}-{inv}", "StockCode": f"S{rng.integers(0, 30)}", "Description": "item",
                             "Quantity": int(rng.integers(1, 12)), "InvoiceDate": start + pd.Timedelta(days=day, hours=10),
                             "Price": round(float(rng.uniform(1, 9)), 2), "Customer ID": 10000 + c, "Country": "UK"})
            day += max(1, int(rng.normal(gap, gap / 3)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "tx.csv"
    synthetic_log().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def service(data_file):
    return TransactionDataService(data_file, prediction_window_days=60, min_history_days=120)


def test_loader_accepts_retail_ii_columns(service):
    tx = service.load_transactions()
    assert {"invoice_no", "customer_id", "unit_price", "invoice_date"} <= set(tx.columns)
    assert tx["customer_id"].str.match(r"^\d+$").all()


def test_features_ignore_the_future(service):
    """Adding transactions on or after D must not change any feature at D."""
    tx = service.load_transactions()
    d = pd.Timestamp("2011-01-15")
    before = service.build_customer_snapshot(tx, d)
    future = tx[tx["invoice_date"] >= d].copy()
    future["quantity"] *= 100
    future["purchase_revenue"] *= 100
    mutated = pd.concat([tx[tx["invoice_date"] < d], future], ignore_index=True)
    after = service.build_customer_snapshot(mutated, d)
    pd.testing.assert_frame_equal(before, after)


def test_label_matches_window(service):
    tx = service.load_transactions()
    d = pd.Timestamp("2011-03-01")
    snap = service.labelled_snapshot(tx, d)
    window = tx[(tx["invoice_date"] >= d) & (tx["invoice_date"] < d + pd.Timedelta(days=60)) & ~tx["is_return"]]
    buyers = set(window["customer_id"])
    for _, row in snap.sample(20, random_state=1).iterrows():
        assert row["churned"] == int(row["customer_id"] not in buyers)


def test_snapshot_windows_do_not_overlap(service):
    plan = service.plan(service.load_transactions())
    w = pd.Timedelta(days=60)
    assert plan.train_dates[-1] + w <= plan.calibration_date
    assert plan.calibration_date + w <= plan.test_date
    assert plan.test_date + w <= plan.data_end


def test_cadence_counts_same_day_orders_as_one_visit(service):
    tx = service.load_transactions()
    one = tx[tx["customer_id"] == tx["customer_id"].iloc[0]].head(1).copy()
    twin = one.copy()
    twin["invoice_no"] = "dup"
    snap = service.build_customer_snapshot(pd.concat([tx, twin]), one["invoice_date"].iloc[0] + pd.Timedelta(days=5))
    row = snap[snap["customer_id"] == one["customer_id"].iloc[0]].iloc[0]
    assert row["avg_days_between_orders"] != 0


def test_app_end_to_end(data_file, tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_DATA_PATH", str(data_file))
    monkeypatch.setenv("CHURN_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    import importlib
    import churn_predictor.config as cfg
    importlib.reload(cfg)
    import churn_predictor
    importlib.reload(churn_predictor)
    app = churn_predictor.create_app()
    for key in ("ARTIFACTS_DIR", "MODEL_PATH", "ANALYSIS_PATH", "PREDICTIONS_PATH", "METRICS_PATH"):
        assert str(tmp_path) in str(app.config[key])
    svc = app.extensions["analysis_service"]
    svc.data_service.min_history = pd.Timedelta(days=120)
    analysis = svc.refresh_analysis(force_retrain=True)
    assert 0.0 <= analysis["model"]["evaluation"]["roc_auc"] <= 1.0
    assert analysis["model"]["selection"].startswith("forward")

    client = app.test_client()
    cid = analysis["customers"][0]["customer_id"]
    for url in ["/", "/customers", "/customers?risk=High", "/customers?q=nomatch", "/methodology",
                f"/customers/{cid}", "/api/summary", "/health"]:
        assert client.get(url).status_code == 200, url
    assert client.get("/customers/does-not-exist").status_code == 404
    assert client.post("/api/retrain").status_code == 403
