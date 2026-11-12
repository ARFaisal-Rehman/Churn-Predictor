# Risk Radar: Customer Churn Intelligence

Risk Radar is a professional-grade churn prediction and customer analytics platform built with Python, Flask, and Scikit-Learn. It transforms raw transaction history into actionable business intelligence by identifying at-risk customers and recommending retention strategies.

## 🚀 Features

- **Automated Feature Engineering**: Aggregates raw transaction data into customer-level behavioral features (RFM, trends, engagement).
- **Advanced ML Pipeline**: Trains and evaluates multiple models (Random Forest, XGBoost, Gradient Boosting) using Stratified K-Fold cross-validation.
- **Dynamic Churn Labeling**: Automatically defines churn based on a configurable observation window.
- **Customer Segmentation**: Uses KMeans and PCA to group customers into "Core", "Watchlist", and "At Risk" segments.
- **Interactive Dashboard**: Modern "Modern Retro" UI for high-level monitoring and deep-dive customer profiling.
- **Explainable AI**: Provides top risk indicators for every customer to guide retention efforts.

## 📁 Project Structure

```
churn predictor/
├── app.py                  # Thin entry point
├── churn_predictor/        # Main application package
│   ├── services/           # Business logic & ML services
│   │   ├── data_service.py # Data loading & feature engineering
│   │   ├── model_service.py# ML training & segmentation
│   │   └── analysis_service.py # Analysis orchestration
│   ├── config.py           # Configuration management
│   ├── routes.py           # Web routes
│   └── __init__.py         # App factory
├── static/                 # CSS, JS, and Images
├── templates/              # Jinja2 HTML templates
├── artifacts/              # Trained models and cached analysis
└── data.csv                # Raw transaction dataset
```

## 🛠️ Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will automatically detect `data.csv`, perform feature engineering, and train the best model on first run.

3. **Retrain Anytime**:
   Use the "Retrain Model" button on the dashboard to refresh the analysis if the dataset changes.

## 📊 ML Methodology

Risk Radar uses a **Look-ahead Window** approach:
1. **Observation Date**: The point in time used to calculate behavioral features.
2. **Prediction Window**: A period after the observation date (default: 60 days).
3. **Target**: A customer is labeled as "churned" if they had activity before the observation date but zero activity during the prediction window.

This ensures the model learns patterns that actually lead to churn in a real business context.

## 🎨 UI Design

The interface follows a **Modern Retro** aesthetic, combining bold typography with clean data visualizations powered by Chart.js. It is designed for clarity, prioritizing high-risk customers and actionable revenue metrics.

## 📈 Results (UCI Online Retail, 60-day prediction window)

| Model | CV ROC-AUC | Test ROC-AUC | F1 |
|---|---|---|---|
| Gradient Boosting | 0.747 | **0.746** | **0.719** |
| Random Forest | 0.754 | 0.742 | 0.701 |
| Logistic Regression | 0.748 | 0.738 | 0.710 |
| XGBoost | 0.737 | 0.724 | 0.691 |

Features are built only from transactions before the observation date and the label comes from the window after it, so there is no target leakage.

## 📦 Data
`data.csv` is not committed (45 MB). Download **Online Retail** from the UCI ML Repository (https://archive.ics.uci.edu/dataset/352/online+retail), export to CSV and place it in the project root as `data.csv`.
