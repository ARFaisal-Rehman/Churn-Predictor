import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, silhouette_score
)


def evaluate_classifier(model, X_test, y_test, proba_ok: bool = True) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    if proba_ok:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = round(float(roc_auc_score(y_test, y_prob)), 4)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        except Exception:
            metrics['roc_auc'] = None
    return metrics


def run_classification(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # Decision Tree with grid search
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = {'max_depth': [3, 5, 8, None], 'min_samples_split': [2, 5, 10]}
    dt_cv = GridSearchCV(dt, dt_grid, cv=3, n_jobs=-1)
    dt_cv.fit(X_train, y_train)
    results['decision_tree'] = {
        'best_params': dt_cv.best_params_,
        'metrics': evaluate_classifier(dt_cv.best_estimator_, X_test, y_test)
    }

    # Naive Bayes (Gaussian)
    nb = GaussianNB()
    nb.fit(X_train.toarray() if hasattr(X_train, 'toarray') else X_train, y_train)
    results['naive_bayes'] = {
        'metrics': evaluate_classifier(nb, X_test.toarray() if hasattr(X_test, 'toarray') else X_test, y_test)
    }

    # KNN with k selection
    knn = KNeighborsClassifier()
    knn_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    knn_cv = GridSearchCV(knn, knn_grid, cv=3, n_jobs=-1)
    knn_cv.fit(X_train, y_train)
    results['knn'] = {
        'best_params': knn_cv.best_params_,
        'metrics': evaluate_classifier(knn_cv.best_estimator_, X_test, y_test)
    }

    return results


def run_clustering(X: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return {
        'labels': labels.tolist(),
        'centers': km.cluster_centers_.tolist(),
        'silhouette': round(float(sil), 4) if sil is not None else None,
        'pca_coords': coords.tolist()
    }

