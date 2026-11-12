"""Transaction loading and leak-free snapshot construction.

A *snapshot* at date D describes every customer who bought something in the
year before D, using only transactions strictly before D. A *labelled*
snapshot adds `churned = 1` when that customer makes no purchase in
[D, D + window). Snapshots are what the model trains and is tested on, so the
only rule that matters is: features never see anything at or after D.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Column names in UCI Online Retail (id 352) and Online Retail II (id 502),
# normalised to lower case without spaces or underscores.
COLUMN_ALIASES = {
    "invoiceno": "invoice_no", "invoice": "invoice_no",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "unitprice": "unit_price", "price": "unit_price",
    "customerid": "customer_id",
    "country": "country",
}

# Stock codes that are fees or adjustments, not products.
NON_PRODUCT_CODES = {"POST", "D", "M", "DOT", "CRUK", "C2", "BANK CHARGES", "AMAZONFEE",
                     "ADJUST", "ADJUST2", "B", "S", "PADS", "TEST001", "TEST002"}


@dataclass
class SnapshotPlan:
    data_start: pd.Timestamp
    data_end: pd.Timestamp          # exclusive: midnight after the last transaction day
    train_dates: List[pd.Timestamp]
    calibration_date: pd.Timestamp
    test_date: pd.Timestamp
    scoring_date: pd.Timestamp


@dataclass
class PreparedDatasets:
    transactions: pd.DataFrame
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    scoring: pd.DataFrame
    plan: SnapshotPlan
    cleaning: Dict[str, int] = field(default_factory=dict)


class TransactionDataService:
    def __init__(self, data_path, prediction_window_days: int = 60, active_lookback_days: int = 365,
                 snapshot_step_days: int = 30, min_history_days: int = 180):
        self.data_path = Path(data_path)
        self.window = pd.Timedelta(days=prediction_window_days)
        self.lookback = pd.Timedelta(days=active_lookback_days)
        self.step = pd.Timedelta(days=snapshot_step_days)
        self.min_history = pd.Timedelta(days=min_history_days)
        self.cleaning: Dict[str, int] = {}

    # ------------------------------------------------------------------ load
    def _read_raw(self) -> pd.DataFrame:
        if self.data_path.suffix.lower() in {".xlsx", ".xls"}:
            sheets = pd.read_excel(self.data_path, sheet_name=None, dtype=str)
            return pd.concat(sheets.values(), ignore_index=True)
        return pd.read_csv(self.data_path, encoding="latin1", dtype=str)

    def load_transactions(self) -> pd.DataFrame:
        df = self._read_raw()
        df.columns = [str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns]
        df = df.rename(columns=COLUMN_ALIASES)
        missing = set(COLUMN_ALIASES.values()).difference(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        stats = {"raw_rows": int(len(df))}
        df = df.drop_duplicates()
        stats["after_exact_duplicates"] = int(len(df))

        cid = pd.to_numeric(df["customer_id"], errors="coerce")
        df = df.assign(customer_id=cid).dropna(subset=["customer_id"])
        df["customer_id"] = df["customer_id"].astype("int64").astype(str)
        stats["with_customer_id"] = int(len(df))

        df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
        df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
        df["description"] = df["description"].fillna("Unknown").astype(str).str.strip()
        df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        df = df.dropna(subset=["invoice_date", "quantity", "unit_price"])

        df = df[~df["stock_code"].isin(NON_PRODUCT_CODES)]
        df = df[df["unit_price"] > 0]
        stats["product_rows_with_price"] = int(len(df))

        df["is_return"] = (df["quantity"] < 0) | df["invoice_no"].str.startswith("C", na=False)
        line_value = df["quantity"] * df["unit_price"]
        df["purchase_revenue"] = np.where(df["is_return"], 0.0, line_value.clip(lower=0.0))
        df["return_value"] = np.where(df["is_return"], line_value.abs(), 0.0)
        df["purchase_quantity"] = np.where(df["is_return"], 0.0, df["quantity"].clip(lower=0.0))
        df["return_quantity"] = np.where(df["is_return"], df["quantity"].clip(upper=0.0).abs(), 0.0)

        stats["purchase_rows"] = int((~df["is_return"]).sum())
        stats["return_rows"] = int(df["is_return"].sum())
        stats["customers"] = int(df["customer_id"].nunique())
        self.cleaning = stats
        return df.sort_values("invoice_date").reset_index(drop=True)

    # ------------------------------------------------------------- snapshots
    def plan(self, transactions: pd.DataFrame, n_train_snapshots: int | None = None) -> SnapshotPlan:
        start = transactions["invoice_date"].min().normalize()
        end = transactions["invoice_date"].max().normalize() + pd.Timedelta(days=1)
        test_date = end - self.window
        calibration_date = test_date - self.window
        latest_train = calibration_date - self.window
        train_dates = []
        d = latest_train
        while d - start >= self.min_history:
            train_dates.append(d)
            d -= self.step
            if n_train_snapshots and len(train_dates) >= n_train_snapshots:
                break
        if not train_dates:
            raise ValueError("Not enough history for any training snapshot; need more data.")
        return SnapshotPlan(start, end, sorted(train_dates), calibration_date, test_date, end)

    def prepare_datasets(self, n_train_snapshots: int | None = None) -> PreparedDatasets:
        tx = self.load_transactions()
        plan = self.plan(tx, n_train_snapshots)
        train = pd.concat([self.labelled_snapshot(tx, d) for d in plan.train_dates], ignore_index=True)
        return PreparedDatasets(
            transactions=tx,
            train=train,
            calibration=self.labelled_snapshot(tx, plan.calibration_date),
            test=self.labelled_snapshot(tx, plan.test_date),
            scoring=self.build_customer_snapshot(tx, plan.scoring_date),
            plan=plan,
            cleaning=dict(self.cleaning),
        )

    def labelled_snapshot(self, transactions: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
        frame = self.build_customer_snapshot(transactions, snapshot_date)
        window = transactions[(transactions["invoice_date"] >= snapshot_date)
                              & (transactions["invoice_date"] < snapshot_date + self.window)
                              & (~transactions["is_return"])]
        buyers = set(window["customer_id"].unique())
        frame["churned"] = (~frame["customer_id"].isin(buyers)).astype(int)
        return frame

    def build_customer_snapshot(self, transactions: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
        """Features for customers active in the year before `snapshot_date`, from data before it only."""
        hist = transactions[transactions["invoice_date"] < snapshot_date]
        purchases = hist[~hist["is_return"]]
        active = purchases.loc[purchases["invoice_date"] >= snapshot_date - self.lookback, "customer_id"].unique()
        hist = hist[hist["customer_id"].isin(active)]
        purchases = purchases[purchases["customer_id"].isin(active)]
        if purchases.empty:
            raise ValueError(f"No active customers before {snapshot_date.date()}.")

        invoices = (purchases.groupby(["customer_id", "invoice_no"])
                    .agg(invoice_date=("invoice_date", "max"), order_value=("purchase_revenue", "sum"),
                         items=("purchase_quantity", "sum"), distinct_products=("stock_code", "nunique"))
                    .reset_index())

        customer = (purchases.groupby("customer_id")
                    .agg(first_purchase=("invoice_date", "min"), last_purchase=("invoice_date", "max"),
                         total_revenue=("purchase_revenue", "sum"), avg_unit_price=("unit_price", "mean"),
                         unique_products=("stock_code", "nunique"), countries_served=("country", "nunique"))
                    .reset_index())
        orders = (invoices.groupby("customer_id")
                  .agg(order_count=("invoice_no", "nunique"), avg_order_value=("order_value", "mean"),
                       median_order_value=("order_value", "median"), max_order_value=("order_value", "max"),
                       avg_items_per_order=("items", "mean"), total_items=("items", "sum"),
                       avg_distinct_products=("distinct_products", "mean"))
                  .reset_index())
        customer = customer.merge(orders, on="customer_id", how="left")

        returns = hist[hist["is_return"]]
        ret = (returns.groupby("customer_id")
               .agg(return_count=("invoice_no", "nunique"), return_value=("return_value", "sum"),
                    returned_items=("return_quantity", "sum")))
        customer = customer.merge(ret, how="left", left_on="customer_id", right_index=True)

        for window in (30, 60, 90, 180):
            cutoff = snapshot_date - pd.Timedelta(days=window)
            recent = purchases[purchases["invoice_date"] >= cutoff]
            recent_orders = invoices[invoices["invoice_date"] >= cutoff]
            a = recent.groupby("customer_id").agg(**{
                f"revenue_last_{window}d": ("purchase_revenue", "sum"),
                f"products_last_{window}d": ("stock_code", "nunique")})
            b = recent_orders.groupby("customer_id").agg(**{f"orders_last_{window}d": ("invoice_no", "nunique")})
            customer = customer.merge(a, how="left", left_on="customer_id", right_index=True)
            customer = customer.merge(b, how="left", left_on="customer_id", right_index=True)

        # Rhythm is measured between distinct purchase days: several invoices on one day are one visit.
        visits = (invoices.assign(day=invoices["invoice_date"].dt.normalize())
                  .drop_duplicates(["customer_id", "day"]).sort_values(["customer_id", "day"]))
        visits["gap_days"] = visits.groupby("customer_id")["day"].diff().dt.days
        cadence = visits.groupby("customer_id").agg(avg_days_between_orders=("gap_days", "mean"),
                                                     std_days_between_orders=("gap_days", "std"),
                                                     purchase_days=("day", "nunique"))
        customer = customer.merge(cadence, how="left", left_on="customer_id", right_index=True)

        days = lambda s: (snapshot_date - s).dt.total_seconds() / 86400.0
        customer["tenure_days"] = days(customer["first_purchase"]).clip(lower=1).round(1)
        customer["recency_days"] = days(customer["last_purchase"]).clip(lower=0).round(1)
        # How overdue the customer is relative to their own rhythm (1.0 = right on schedule).
        customer["recency_vs_cadence"] = customer["recency_days"] / customer["avg_days_between_orders"].clip(lower=1)
        customer["orders_per_month"] = customer["order_count"] / (customer["tenure_days"] / 30.0).clip(lower=1.0)
        customer["revenue_per_day"] = customer["total_revenue"] / customer["tenure_days"].clip(lower=1.0)
        customer["return_rate"] = customer["return_count"] / customer["order_count"].replace(0, np.nan)
        customer["return_value_ratio"] = customer["return_value"] / customer["total_revenue"].replace(0, np.nan)
        customer["revenue_trend_90_180"] = customer["revenue_last_90d"] / (
            customer["revenue_last_180d"] - customer["revenue_last_90d"]).replace(0, np.nan)
        # Expected revenue in a window of the prediction length, from the customer's own history.
        customer["value_per_window"] = customer["revenue_per_day"] * self.window.days
        # Season of the prediction window: this retailer has a strong pre-Christmas peak, so the
        # same behaviour means different churn odds in October and in February.
        angle = 2 * np.pi * (snapshot_date.dayofyear / 365.25)
        customer["season_sin"] = np.sin(angle)
        customer["season_cos"] = np.cos(angle)

        fill_zero = [c for c in customer.columns if c.startswith(("revenue_last", "orders_last", "products_last",
                                                                  "return_"))] + ["returned_items"]
        customer[fill_zero] = customer[fill_zero].fillna(0.0)
        customer["snapshot_date"] = snapshot_date.strftime("%Y-%m-%d")
        return customer.drop(columns=["first_purchase", "last_purchase"]).sort_values("customer_id").reset_index(drop=True)

    @staticmethod
    def build_overview(transactions: pd.DataFrame) -> Dict[str, float]:
        purchases = transactions.loc[~transactions["is_return"]]
        revenue = float(purchases["purchase_revenue"].sum())
        orders = int(purchases["invoice_no"].nunique())
        return {
            "transaction_rows": int(len(transactions)),
            "customer_count": int(purchases["customer_id"].nunique()),
            "order_count": orders,
            "total_revenue": round(revenue, 2),
            "average_order_value": round(revenue / orders, 2) if orders else 0.0,
            "first_date": transactions["invoice_date"].min().strftime("%Y-%m-%d"),
            "last_date": transactions["invoice_date"].max().strftime("%Y-%m-%d"),
        }
