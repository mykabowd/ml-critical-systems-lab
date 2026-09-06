"""Tests des fonctions d'evaluation (metrics.py) sur des donnees synthetiques
simples, ou le resultat attendu peut etre calcule a la main.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_critical_systems_lab.metrics import (
    anomaly_detection_auc,
    classification_metrics,
    confusion_matrix_df,
    regression_metrics,
    rmse,
    spearman_correlation,
)


def test_rmse_perfect_predictions():
    y_true = [1, 2, 3, 4]
    assert rmse(y_true, y_true) == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    # erreur constante de 1 -> RMSE = 1
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_regression_metrics_known_values():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 18, 33, 37])
    metrics = regression_metrics(y_true, y_pred)

    errors = y_true - y_pred
    expected_rmse = float(np.sqrt(np.mean(errors ** 2)))
    expected_mae = float(np.mean(np.abs(errors)))

    assert metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics["mae"] == pytest.approx(expected_mae)
    assert metrics["r2"] <= 1.0


def test_classification_metrics_known_confusion():
    # 4 vrais positifs, 1 faux positif, 1 faux negatif, 4 vrais negatifs.
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

    metrics = classification_metrics(y_true, y_pred)

    # precision = TP / (TP + FP) = 4 / (4 + 1) = 0.8
    assert metrics["precision"] == pytest.approx(0.8)
    # recall = TP / (TP + FN) = 4 / (4 + 1) = 0.8
    assert metrics["recall"] == pytest.approx(0.8)
    assert metrics["f1"] == pytest.approx(0.8)
    # accuracy = 8/10
    assert metrics["accuracy"] == pytest.approx(0.8)


def test_confusion_matrix_df_shape_and_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    cm = confusion_matrix_df(y_true, y_pred)

    assert cm.shape == (2, 2)
    assert cm.loc["vrai_0", "predit_0"] == 1
    assert cm.loc["vrai_0", "predit_1"] == 1
    assert cm.loc["vrai_1", "predit_1"] == 2


def test_anomaly_detection_auc_perfect_separation():
    # scores parfaitement separes selon le label -> AUC = 1.0
    scores = [0.1, 0.2, 0.9, 0.95]
    labels = [0, 0, 1, 1]
    assert anomaly_detection_auc(scores, labels) == pytest.approx(1.0)


def test_anomaly_detection_auc_requires_two_classes():
    with pytest.raises(ValueError):
        anomaly_detection_auc([0.1, 0.2, 0.3], [0, 0, 0])


def test_spearman_correlation_monotonic():
    a = [1, 2, 3, 4, 5]
    b = [10, 20, 30, 40, 50]
    assert spearman_correlation(a, b) == pytest.approx(1.0)

    b_decreasing = [50, 40, 30, 20, 10]
    assert spearman_correlation(a, b_decreasing) == pytest.approx(-1.0)
