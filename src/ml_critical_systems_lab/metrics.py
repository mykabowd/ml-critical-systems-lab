"""Fonctions d'evaluation partagees (regression, classification, detection
d'anomalies). Isolees dans un module dedie pour etre testees independamment
de l'entrainement des modeles (rapide, deterministe, sur donnees synthetiques).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE et R^2 pour une tache de regression."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Precision, recall, F1 (classe positive = 1) et accuracy."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def confusion_matrix_df(y_true, y_pred, labels: tuple = (0, 1)) -> pd.DataFrame:
    """Matrice de confusion sous forme de DataFrame lisible."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    index = [f"vrai_{lbl}" for lbl in labels]
    columns = [f"predit_{lbl}" for lbl in labels]
    return pd.DataFrame(cm, index=index, columns=columns)


def anomaly_detection_auc(anomaly_scores, is_near_failure) -> float:
    """AUC-ROC : capacite du score d'anomalie a discriminer les etats
    "proche de la panne" (label=1) des etats normaux (label=0).

    Un score d'anomalie utile doit etre plus eleve pres de la panne : cette
    metrique quantifie directement cette hypothese sans etiquette de
    supervision explicite pour l'entrainement du detecteur (seulement pour
    l'evaluation).
    """
    anomaly_scores = np.asarray(anomaly_scores)
    is_near_failure = np.asarray(is_near_failure)
    if len(np.unique(is_near_failure)) < 2:
        raise ValueError(
            "is_near_failure doit contenir au moins deux classes (0 et 1) pour calculer un AUC."
        )
    return float(roc_auc_score(is_near_failure, anomaly_scores))


def spearman_correlation(a, b) -> float:
    """Correlation de Spearman entre deux vecteurs (rangs), sans dependance
    a scipy : pandas calcule les rangs puis une correlation de Pearson.
    """
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
