<div align="center">

```
██████╗ ██╗███████╗██╗  ██╗    ██████╗  █████╗ ██████╗  █████╗ ██████╗
██╔══██╗██║██╔════╝██║ ██╔╝    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗
██████╔╝██║███████╗█████╔╝     ██████╔╝███████║██║  ██║███████║██████╔╝
██╔══██╗██║╚════██║██╔═██╗     ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗
██║  ██║██║███████║██║  ██╗    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║
╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
```

### Customer Churn Intelligence for Retail: Who Is About to Leave, Why, and What to Do

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge)](https://xgboost.ai)
[![pandas](https://img.shields.io/badge/pandas-Feature_Eng-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

**Test ROC-AUC 0.746 · 4,339 customers scored · ~GBP 985K revenue flagged at risk · zero target leakage**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Methodology](#ml-methodology)
- [Results](#results)
- [Customer Segments](#customer-segments)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Challenges and Solutions](#challenges-and-solutions)
- [Author](#author)

---

## Overview

Risk Radar turns a raw retail transaction log into a ranked list of customers who are likely to stop buying, the signals behind each prediction, and a recommended retention action. It is built on the UCI **Online Retail** dataset (406,829 valid transactions, 18,536 orders, 4,339 customers, Dec 2010 to Dec 2011).

### What makes it different from a typical churn notebook?

| Aspect | Typical Student Churn Project | Risk Radar |
|--------|-------------------------------|-----------|
| Churn label | `Recency > N days`, with Recency also used as a feature (leaks the answer) | **Look-ahead window**: features before the observation date, label from the 60 days after it |
| Features | 3 to 5 RFM columns | **40+ behavioural features**: rolling 30/60/90-day windows, order cadence, returns, product diversity |
| Model selection | One model, one split | **4 models** compared with 5-fold stratified CV, best one picked automatically |
| Explainability | None | **Permutation importance** globally, plus top risk reasons per customer |
| Output | A metric | A **dashboard**: risk tiers, segments, revenue at risk, per-customer profile and action |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                 │
│                                                                      │
│   data.csv (UCI Online Retail)                                       │
│        │                                                             │
│        ▼                                                             │
│   TransactionDataService                                             │
│   ├── clean: drop null IDs, split purchases vs. returns              │
│   ├── observation_date = last_date - 60 days                         │
│   ├── history  = transactions <= observation_date  ──► features      │
│   └── future   = transactions  > observation_date  ──► churn label   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                 │
│                                                                      │
│   ChurnModelService                                                  │
│   ├── ColumnTransformer (impute + scale)                             │
│   ├── LogisticRegression │ RandomForest │ GradientBoosting │ XGBoost │
│   ├── StratifiedKFold (k=5) ──► pick best by CV ROC-AUC              │
│   ├── permutation_importance ──► global drivers                      │
│   └── KMeans (k=3) + PCA ──► Core / Watchlist / At Risk segments     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                             │
│                                                                      │
│   AnalysisService ──► artifacts/ (model bundle, analysis.json)       │
│        │                                                             │
│        ▼                                                             │
│   Flask app factory                                                  │
│   ├── /                    dashboard: KPIs, risk tiers, top risks    │
│   ├── /analytics           model comparison, drivers, segment PCA    │
│   ├── /customers/<id>      profile, risk reasons, suggested action   │
│   └── /api/*               JSON summary, retrain trigger, health     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ML Methodology

### Leak-free labelling

```
 Dec 2010                          10 Oct 2011            09 Dec 2011
    │                                   │                      │
    ├────────── HISTORY ────────────────┼───── 60-DAY WINDOW ──┤
    │  every feature is computed here   │  label: did they buy?│
    │                                   │  no purchase = churn │
                              observation_date
```

The model never sees anything from the prediction window, so the score reflects what it would do on real, future data.

<details>
<summary><b>Feature groups (40+ features)</b></summary>

| Group | Examples |
|-------|----------|
| RFM core | `recency_days`, `order_count`, `total_revenue`, `tenure_days`, `recency_ratio` |
| Order shape | `avg_order_value`, `median_order_value`, `max_order_value`, `avg_items_per_order` |
| Rolling windows | `revenue_last_30d/60d/90d`, `orders_last_30d/60d/90d`, `products_last_60d/90d` |
| Cadence | `avg_days_between_orders`, `std_days_between_orders` |
| Returns | `return_count`, `return_value`, `returned_items` |
| Breadth | `unique_products`, `unique_descriptions`, `countries_served` |
| Scores | `engagement_score`, `monetary_score` |

</details>

<details>
<summary><b>Model selection</b></summary>

Each candidate is wrapped in a scikit-learn `Pipeline` with the same preprocessor, scored with 5-fold `StratifiedKFold` on the training split, then evaluated once on a held-out 25% test split. The model with the best CV ROC-AUC is persisted with `joblib` and used to score every customer.

</details>

---

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |
|-------|-----------:|-------------:|----------:|-------:|---:|
| **Gradient Boosting** (selected) | 0.747 | **0.746** | 0.675 | 0.770 | **0.719** |
| Random Forest | 0.754 | 0.742 | 0.663 | 0.743 | 0.701 |
| Logistic Regression | 0.748 | 0.738 | 0.664 | 0.764 | 0.710 |
| XGBoost | 0.737 | 0.724 | 0.657 | 0.728 | 0.691 |

Training rows: 3,721 customers active before the observation date · churn rate 51.8%.

### Top churn drivers (permutation importance)

```
order_count               ████████████████████  0.0090
products_last_60d         ██████████████████    0.0082
recency_ratio             ██████████████        0.0063
products_last_90d         ██████████████        0.0063
unique_descriptions       ████████████          0.0055
engagement_score          ███████████           0.0051
max_order_value           ███████████           0.0048
```

Customers who order rarely and have stopped exploring new products in the last two months are the clearest churn signal.

---

## Customer Segments

| Segment | Customers | Avg. Churn Risk | Avg. Revenue (GBP) | Avg. Orders | Suggested Play |
|---------|----------:|----------------:|-------------:|------------:|----------------|
| **Core** | 6 | 18% | 180,072 | 98.8 | Account management, protect at all costs |
| **Watchlist** | 3,077 | 39% | 2,276 | 5.2 | Cross-sell and loyalty nudges |
| **At Risk** | 1,256 | 72% | 660 | 1.7 | Win-back offer within the next cycle |

Risk tiers across all scored customers: **1,494 high · 1,384 medium · 1,461 low**, with an estimated **GBP 984,650** of revenue tied to high-risk customers.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard: KPIs, risk tiers, top at-risk customers |
| `GET` | `/analytics` | Model comparison, feature importance, segment PCA |
| `GET` | `/customers/<customer_id>` | Customer profile, top risk reasons, recommended action |
| `GET` | `/api/summary` | Full analysis as JSON |
| `POST` | `/api/retrain` | Rebuild features and retrain all models |
| `GET` | `/health` | Health check |

---

## Project Structure

```
risk-radar/
├── app.py                          entry point
├── churn_predictor/
│   ├── __init__.py                 Flask app factory
│   ├── config.py                   paths, window size, seeds (env-overridable)
│   ├── routes.py                   pages and JSON API
│   └── services/
│       ├── data_service.py         cleaning, look-ahead split, feature engineering
│       ├── model_service.py        pipelines, CV, selection, segmentation, explanations
│       └── analysis_service.py     orchestration and caching to artifacts/
├── templates/                      dashboard, analytics, customer profile
├── static/                         CSS and JS (charts)
└── requirements.txt
```

---

## Getting Started

```bash
# 1. Install
pip install -r requirements.txt

# 2. Data (not committed, 45 MB)
#    Download "Online Retail" from https://archive.ics.uci.edu/dataset/352/online+retail
#    export the sheet to CSV and save it as data.csv in the project root

# 3. Run (first launch trains the models, about a minute)
python app.py
# open http://localhost:5000
```

Configuration via environment variables: `PREDICTION_WINDOW_DAYS` (default 60), `TEST_SIZE` (0.25), `RANDOM_STATE` (42), `CHURN_DATA_PATH`.

---

## Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| First version scored ~100% because churn was defined from Recency, which was also a feature | Rebuilt labelling around an observation date: features from history only, label from the following 60 days |
| Raw `first_purchase` / `last_purchase` timestamps crashed the numeric pipeline | Feature selector now drops datetime columns; their signal is already captured by recency and tenure |
| Cancelled invoices (negative quantities) distorted revenue | Split into separate purchase and return streams; returns became their own features |
| Very skewed revenue (a few customers spend 100x the median) | Scaled inside the pipeline and segmented with KMeans on standardised features |

---

## Author

**Abdul Rehman Faisal** · BS Data Science, GIKI (2027)
[GitHub](https://github.com/ARFaisal-Rehman) · [LinkedIn](https://linkedin.com/in/abdul-rehman-faisal-2a680b297)

---

<div align="center">

**Built with Python, scikit-learn and Flask · Data: UCI Machine Learning Repository**

</div>
