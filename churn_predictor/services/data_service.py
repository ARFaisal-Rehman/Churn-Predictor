from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class PreparedDatasets:
    transactions: pd.DataFrame
    training_frame: pd.DataFrame
    scoring_frame: pd.DataFrame
    reference_date: pd.Timestamp
    observation_date: pd.Timestamp


class TransactionDataService:
    REQUIRED_COLUMNS = {
        "invoiceno",
        "stockcode",
        "description",
        "quantity",
        "invoicedate",
        "unitprice",
        "customerid",
        "country",
    }

    def __init__(self, data_path, prediction_window_days: int = 60):
        self.data_path = data_path
        self.prediction_window_days = prediction_window_days

    def load_transactions(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, encoding="latin1")
        df.columns = [col.strip().lower() for col in df.columns]

        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        # Map to snake_case for consistent internal use
        df = df.rename(columns={
            "invoiceno": "invoice_no",
            "stockcode": "stock_code",
            "invoicedate": "invoice_date",
            "unitprice": "unit_price",
            "customerid": "customer_id"
        })

        df = df.copy()
        df["customer_id"] = df["customer_id"].astype(str).str.strip()
        df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
        df["stock_code"] = df["stock_code"].astype(str).str.strip()
        df["description"] = df["description"].fillna("Unknown").astype(str).str.strip()
        df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

        df = df.dropna(subset=["customer_id", "invoice_date", "quantity", "unit_price"])
        df = df[df["customer_id"].ne("") & df["customer_id"].ne("nan")]

        df["is_return"] = (df["quantity"] < 0) | df["invoice_no"].str.startswith("C", na=False)
        df["gross_revenue"] = df["quantity"] * df["unit_price"]
        df["purchase_revenue"] = np.where(df["is_return"], 0.0, np.clip(df["gross_revenue"], 0.0, None))
        df["return_value"] = np.where(df["is_return"], np.abs(df["gross_revenue"]), 0.0)
        df["purchase_quantity"] = np.where(df["is_return"], 0.0, np.clip(df["quantity"], 0.0, None))
        df["return_quantity"] = np.where(df["is_return"], np.abs(np.clip(df["quantity"], None, 0.0)), 0.0)
        df["invoice_day"] = df["invoice_date"].dt.floor("D")

        return df.sort_values("invoice_date").reset_index(drop=True)

    def prepare_datasets(self) -> PreparedDatasets:
        transactions = self.load_transactions()
        reference_date = transactions["invoice_date"].max().floor("D")
        observation_date = reference_date - pd.Timedelta(days=self.prediction_window_days)

        history = transactions[transactions["invoice_date"] <= observation_date].copy()
        future = transactions[transactions["invoice_date"] > observation_date].copy()

        training_frame = self.build_customer_snapshot(history, observation_date)
        future_activity = (
            future.loc[~future["is_return"]]
            .groupby("customer_id")
            .size()
            .rename("future_orders")
        )
        training_frame = training_frame.merge(
            future_activity,
            how="left",
            left_on="customer_id",
            right_index=True,
        )
        training_frame["future_orders"] = training_frame["future_orders"].fillna(0).astype(int)
        training_frame["churned"] = (training_frame["future_orders"] == 0).astype(int)
        training_frame = training_frame.drop(columns=["future_orders"])

        scoring_frame = self.build_customer_snapshot(transactions, reference_date)

        return PreparedDatasets(
            transactions=transactions,
            training_frame=training_frame,
            scoring_frame=scoring_frame,
            reference_date=reference_date,
            observation_date=observation_date,
        )

    def build_customer_snapshot(
        self,
        transactions: pd.DataFrame,
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        purchase_rows = transactions.loc[~transactions["is_return"]].copy()
        if purchase_rows.empty:
            raise ValueError("No valid purchase rows found in the dataset.")

        invoice_agg = (
            purchase_rows.groupby(["customer_id", "invoice_no"])
            .agg(
                invoice_date=("invoice_date", "max"),
                order_value=("purchase_revenue", "sum"),
                items=("purchase_quantity", "sum"),
                distinct_products=("stock_code", "nunique"),
            )
            .reset_index()
        )

        customer = (
            purchase_rows.groupby("customer_id")
            .agg(
                first_purchase=("invoice_date", "min"),
                last_purchase=("invoice_date", "max"),
                total_revenue=("purchase_revenue", "sum"),
                avg_unit_price=("unit_price", "mean"),
                unique_products=("stock_code", "nunique"),
                unique_descriptions=("description", "nunique"),
                countries_served=("country", "nunique"),
            )
            .reset_index()
        )

        orders = (
            invoice_agg.groupby("customer_id")
            .agg(
                order_count=("invoice_no", "nunique"),
                avg_order_value=("order_value", "mean"),
                median_order_value=("order_value", "median"),
                max_order_value=("order_value", "max"),
                avg_items_per_order=("items", "mean"),
                total_items=("items", "sum"),
                avg_distinct_products=("distinct_products", "mean"),
            )
            .reset_index()
        )

        customer = customer.merge(orders, on="customer_id", how="left")

        return_rows = transactions.loc[transactions["is_return"]].copy()
        if not return_rows.empty:
            returns = (
                return_rows.groupby("customer_id")
                .agg(
                    return_count=("invoice_no", "nunique"),
                    return_value=("return_value", "sum"),
                    returned_items=("return_quantity", "sum"),
                )
                .reset_index()
            )
            customer = customer.merge(returns, on="customer_id", how="left")

        customer[["return_count", "return_value", "returned_items"]] = customer[
            ["return_count", "return_value", "returned_items"]
        ].fillna(0.0)

        recent_windows: List[int] = [30, 60, 90]
        for window in recent_windows:
            cutoff = reference_date - pd.Timedelta(days=window)
            recent = purchase_rows.loc[purchase_rows["invoice_date"] > cutoff]
            recent_orders = invoice_agg.loc[invoice_agg["invoice_date"] > cutoff]

            recent_features = recent.groupby("customer_id").agg(
                **{
                    f"revenue_last_{window}d": ("purchase_revenue", "sum"),
                    f"items_last_{window}d": ("purchase_quantity", "sum"),
                    f"products_last_{window}d": ("stock_code", "nunique"),
                }
            )

            recent_order_features = recent_orders.groupby("customer_id").agg(
                **{
                    f"orders_last_{window}d": ("invoice_no", "nunique"),
                    f"avg_order_value_last_{window}d": ("order_value", "mean"),
                }
            )

            customer = customer.merge(
                recent_features,
                how="left",
                left_on="customer_id",
                right_index=True,
            ).merge(
                recent_order_features,
                how="left",
                left_on="customer_id",
                right_index=True,
            )

        invoice_agg = invoice_agg.sort_values(["customer_id", "invoice_date"])
        invoice_agg["days_between_orders"] = invoice_agg.groupby("customer_id")["invoice_date"].diff().dt.days
        cadence = invoice_agg.groupby("customer_id").agg(
            avg_days_between_orders=("days_between_orders", "mean"),
            std_days_between_orders=("days_between_orders", "std"),
        )
        customer = customer.merge(cadence, how="left", left_on="customer_id", right_index=True)

        customer["tenure_days"] = (reference_date - customer["first_purchase"]).dt.days.clip(lower=1)
        customer["recency_days"] = (reference_date - customer["last_purchase"]).dt.days.clip(lower=0)
        customer["order_frequency_30d"] = customer["order_count"] / (customer["tenure_days"] / 30.0).clip(lower=1.0)
        customer["revenue_per_day"] = customer["total_revenue"] / customer["tenure_days"].clip(lower=1.0)
        customer["return_rate"] = customer["return_count"] / customer["order_count"].replace(0, np.nan)
        customer["return_value_ratio"] = customer["return_value"] / customer["total_revenue"].replace(0, np.nan)
        customer["recency_ratio"] = customer["recency_days"] / customer["tenure_days"].clip(lower=1.0)
        customer["recent_revenue_share"] = customer["revenue_last_30d"] / customer["total_revenue"].replace(0, np.nan)
        customer["recent_order_share"] = customer["orders_last_30d"] / customer["order_count"].replace(0, np.nan)
        customer["monetary_score"] = np.log1p(customer["total_revenue"])
        customer["engagement_score"] = (
            customer["orders_last_90d"].fillna(0) * 0.5
            + customer["products_last_90d"].fillna(0) * 0.3
            + customer["revenue_last_90d"].fillna(0) * 0.2 / 100.0
        )

        customer = customer.fillna(0.0)
        customer["snapshot_date"] = reference_date.strftime("%Y-%m-%d")
        return customer.sort_values("customer_id").reset_index(drop=True)

    def build_overview(self, transactions: pd.DataFrame) -> Dict[str, float]:
        purchase_rows = transactions.loc[~transactions["is_return"]].copy()
        total_revenue = float(purchase_rows["purchase_revenue"].sum())
        order_count = int(purchase_rows["invoice_no"].nunique())
        customer_count = int(purchase_rows["customer_id"].nunique())

        return {
            "transaction_count": int(len(transactions)),
            "customer_count": customer_count,
            "order_count": order_count,
            "total_revenue": round(total_revenue, 2),
            "average_order_value": round(total_revenue / order_count, 2) if order_count else 0.0,
        }
