"""Modeles supervises :
(a) regression pour predire le RUL (Random Forest vs regression lineaire) ;
(b) classification binaire "maintenance necessaire bientot" (RUL < seuil),
    avec Logistic Regression et Random Forest Classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml_critical_systems_lab.config import MAINTENANCE_SOON_THRESHOLD, RANDOM_STATE
from ml_critical_systems_lab.metrics import classification_metrics, regression_metrics


@dataclass
class RegressionResults:
    models: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass
class ClassificationResults:
    models: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> RegressionResults:
    """Entraine une regression lineaire et un Random Forest Regressor pour
    predire le RUL, et evalue les deux sur le jeu de test (RMSE/MAE/R^2).
    """
    results = RegressionResults()

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    results.models["linear_regression"] = linreg

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.models["random_forest"] = rf

    for name, model in results.models.items():
        preds = model.predict(X_test)
        results.predictions[name] = preds
        results.metrics[name] = regression_metrics(y_test, preds)

    return results


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ClassificationResults:
    """Entraine une Logistic Regression et un Random Forest Classifier pour
    predire le label binaire "maintenance necessaire bientot" (RUL < seuil),
    et evalue les deux sur le jeu de test (precision/recall/F1/accuracy).
    """
    results = ClassificationResults()

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    results.models["logistic_regression"] = logreg

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    results.models["random_forest_classifier"] = rf_clf

    for name, model in results.models.items():
        preds = model.predict(X_test)
        results.predictions[name] = preds
        results.metrics[name] = classification_metrics(y_test, preds)

    return results
