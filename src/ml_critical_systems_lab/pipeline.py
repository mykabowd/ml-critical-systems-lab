"""Pipeline de bout en bout : chargement des donnees -> feature engineering
-> entrainement des 3 familles de modeles (supervise, non supervise, deep
learning) -> evaluation avec de vraies metriques -> generation des figures
-> export d'un fichier `reports/results.json` servant de source de verite
pour le rapport technique (aucune metrique n'est ecrite a la main).

Usage :
    python -m ml_critical_systems_lab.pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from ml_critical_systems_lab.config import (
    FIGURES_DIR,
    MAINTENANCE_SOON_THRESHOLD,
    RANDOM_STATE,
    REPORTS_DIR,
    ROLLING_SENSOR_COLS,
    RUL_CAP,
)
from ml_critical_systems_lab.data import load_rul, load_test, load_train
from ml_critical_systems_lab.deep_learning import (
    DEFAULT_SEQUENCE_LENGTH,
    CmapssSequenceDataset,
    LSTMRULRegressor,
    build_last_window_sequences,
    build_training_sequences,
    predict as dl_predict,
    train_lstm,
)
from ml_critical_systems_lab.features import (
    add_maintenance_soon_label,
    add_rolling_features,
    build_feature_columns,
    compute_rul,
    get_last_cycle_per_unit,
    scale_features,
)
from ml_critical_systems_lab.metrics import (
    anomaly_detection_auc,
    classification_metrics,
    confusion_matrix_df,
    regression_metrics,
    spearman_correlation,
)
from ml_critical_systems_lab.supervised import train_classification_models, train_regression_models
from ml_critical_systems_lab.unsupervised import (
    fit_isolation_forest,
    fit_pca,
    isolation_forest_scores,
    pca_reconstruction_error,
)
from ml_critical_systems_lab.visualize import (
    plot_anomaly_score_distribution,
    plot_confusion_matrix,
    plot_rul_scatter,
    plot_sensor_degradation,
    plot_training_curve,
)


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}")


def load_and_prepare_data() -> dict:
    """Charge les donnees brutes, calcule le RUL, ajoute les features
    glissantes et normalise. Retourne un dict avec toutes les tables
    intermediaires necessaires aux etapes suivantes.
    """
    _log("Chargement des donnees brutes (train/test/RUL)...")
    train_raw = load_train()
    test_raw = load_test()
    rul_test = load_rul()

    _log(f"  train: {train_raw.shape}, {train_raw['unit_number'].nunique()} unites")
    _log(f"  test:  {test_raw.shape}, {test_raw['unit_number'].nunique()} unites")

    _log(f"Calcul du label RUL (plafonne a {RUL_CAP} cycles pour l'entrainement)...")
    train_with_rul = compute_rul(train_raw, cap=RUL_CAP)
    train_with_rul = add_maintenance_soon_label(train_with_rul, threshold=MAINTENANCE_SOON_THRESHOLD)

    _log(f"Feature engineering : stats glissantes sur {len(ROLLING_SENSOR_COLS)} capteurs...")
    train_feat = add_rolling_features(train_with_rul)
    test_feat = add_rolling_features(test_raw)

    feature_cols = build_feature_columns()
    _log(f"  {len(feature_cols)} colonnes de features au total.")

    _log("Normalisation (StandardScaler ajuste sur le train uniquement)...")
    scaled = scale_features(train_feat, test_feat, feature_cols)

    # Verite terrain RUL pour le jeu de test : derniere ligne observee de
    # chaque unite de test, associee a RUL_FD001.txt (valeur reelle, non
    # plafonnee : c'est la definition officielle du benchmark C-MAPSS).
    test_last_cycle = get_last_cycle_per_unit(scaled.test)
    test_last_cycle = test_last_cycle.merge(rul_test, on="unit_number", how="left")
    test_last_cycle = add_maintenance_soon_label(test_last_cycle, threshold=MAINTENANCE_SOON_THRESHOLD)

    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_feat": train_feat,
        "test_feat": test_feat,
        "train_scaled": scaled.train,
        "test_scaled": scaled.test,
        "test_last_cycle": test_last_cycle,
        "feature_cols": feature_cols,
        "scaler": scaled.scaler,
    }


def run_supervised(data: dict) -> dict:
    _log("=== Modeles supervises ===")
    feature_cols = data["feature_cols"]
    train_df = data["train_scaled"]
    test_df = data["test_last_cycle"]

    X_train = train_df[feature_cols]
    y_train_reg = train_df["RUL"]
    X_test = test_df[feature_cols]
    y_test_reg = test_df["RUL"]

    _log("Regression (RUL) : LinearRegression vs RandomForestRegressor...")
    reg_results = train_regression_models(X_train, y_train_reg, X_test, y_test_reg)
    for name, m in reg_results.metrics.items():
        _log(f"  {name}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    y_train_clf = train_df["maintenance_soon"]
    y_test_clf = test_df["maintenance_soon"]

    _log("Classification (maintenance_soon) : LogisticRegression vs RandomForestClassifier...")
    clf_results = train_classification_models(X_train, y_train_clf, X_test, y_test_clf)
    for name, m in clf_results.metrics.items():
        _log(
            f"  {name}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"f1={m['f1']:.3f} accuracy={m['accuracy']:.3f}"
        )

    # Figures : scatter RUL predit vs reel (meilleur modele = random_forest),
    # matrice de confusion (meilleur modele = random_forest_classifier).
    plot_rul_scatter(
        y_test_reg,
        reg_results.predictions["random_forest"],
        title="Regression RUL (Random Forest) : predit vs reel (test FD001)",
        save_path=FIGURES_DIR / "regression_rf_scatter.png",
    )
    plot_rul_scatter(
        y_test_reg,
        reg_results.predictions["linear_regression"],
        title="Regression RUL (regression lineaire) : predit vs reel (test FD001)",
        save_path=FIGURES_DIR / "regression_linear_scatter.png",
    )

    cm = confusion_matrix_df(y_test_clf, clf_results.predictions["random_forest_classifier"])
    plot_confusion_matrix(
        cm,
        title="Matrice de confusion - maintenance_soon (Random Forest)",
        save_path=FIGURES_DIR / "classification_confusion_matrix.png",
    )

    return {
        "regression": reg_results.metrics,
        "classification": clf_results.metrics,
        "classification_confusion_matrix": cm.to_dict(),
    }


def run_unsupervised(data: dict) -> dict:
    _log("=== Detection d'anomalies non supervisee ===")
    train_df = data["train_scaled"]
    sensor_feature_cols = ROLLING_SENSOR_COLS  # capteurs bruts scales

    rng = np.random.RandomState(RANDOM_STATE)
    units = train_df["unit_number"].unique()
    rng.shuffle(units)
    n_fit = int(0.8 * len(units))
    fit_units, eval_units = units[:n_fit], units[n_fit:]

    fit_df = train_df[train_df["unit_number"].isin(fit_units)]
    eval_df = train_df[train_df["unit_number"].isin(eval_units)]

    X_fit = fit_df[sensor_feature_cols]
    X_eval = eval_df[sensor_feature_cols]
    near_failure_eval = (eval_df["RUL"] < MAINTENANCE_SOON_THRESHOLD).astype(int)

    _log(f"  fit sur {len(fit_units)} unites, evaluation sur {len(eval_units)} unites (holdout).")

    _log("Isolation Forest...")
    iso_model = fit_isolation_forest(X_fit)
    iso_scores_eval = isolation_forest_scores(iso_model, X_eval)
    iso_auc = anomaly_detection_auc(iso_scores_eval, near_failure_eval)
    iso_corr = spearman_correlation(iso_scores_eval, eval_df["RUL"].to_numpy())
    _log(f"  AUC (score vs proche-panne) = {iso_auc:.3f} ; correlation Spearman(score, RUL) = {iso_corr:.3f}")

    _log("PCA (erreur de reconstruction)...")
    pca_model = fit_pca(X_fit, n_components=5)
    pca_scores_eval = pca_reconstruction_error(pca_model, X_eval)
    pca_auc = anomaly_detection_auc(pca_scores_eval, near_failure_eval)
    pca_corr = spearman_correlation(pca_scores_eval, eval_df["RUL"].to_numpy())
    _log(f"  AUC (score vs proche-panne) = {pca_auc:.3f} ; correlation Spearman(score, RUL) = {pca_corr:.3f}")
    _log(f"  variance expliquee cumulee (5 composantes) = {pca_model.explained_variance_ratio_.sum():.3f}")

    scores_normal = iso_scores_eval[near_failure_eval.to_numpy() == 0]
    scores_near_failure = iso_scores_eval[near_failure_eval.to_numpy() == 1]
    plot_anomaly_score_distribution(
        scores_normal,
        scores_near_failure,
        title="Distribution du score d'anomalie (Isolation Forest) - normal vs proche panne",
        save_path=FIGURES_DIR / "anomaly_score_distribution.png",
    )

    return {
        "isolation_forest": {"auc": iso_auc, "spearman_corr_with_rul": iso_corr},
        "pca_reconstruction": {
            "auc": pca_auc,
            "spearman_corr_with_rul": pca_corr,
            "explained_variance_ratio_cumsum": float(pca_model.explained_variance_ratio_.sum()),
        },
        "n_fit_units": int(len(fit_units)),
        "n_eval_units": int(len(eval_units)),
    }


def run_deep_learning(data: dict) -> dict:
    _log("=== Deep learning (LSTM, sequences temporelles) ===")
    feature_cols = data["feature_cols"]
    train_df = data["train_scaled"]
    test_df = data["test_scaled"]
    rul_test_gt = data["test_last_cycle"][["unit_number", "RUL"]]

    seq_len = DEFAULT_SEQUENCE_LENGTH
    _log(f"Construction des sequences d'entrainement (longueur={seq_len})...")
    X_seq, y_seq, unit_seq = build_training_sequences(train_df, feature_cols, sequence_length=seq_len)
    _log(f"  {X_seq.shape[0]} sequences, forme = {X_seq.shape}")

    dataset = CmapssSequenceDataset(X_seq, y_seq)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    torch.manual_seed(RANDOM_STATE)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    model = LSTMRULRegressor(n_features=len(feature_cols), hidden_size=32, num_layers=1)

    start = time.time()
    history = train_lstm(model, train_ds, val_ds, epochs=15, batch_size=64, lr=1e-3)
    elapsed = time.time() - start
    _log(f"Entrainement termine en {elapsed:.1f}s.")

    plot_training_curve(history.train_loss, history.val_loss, save_path=FIGURES_DIR / "lstm_training_curve.png")

    _log("Evaluation sur le jeu de test officiel (dernier cycle observe de chaque unite)...")
    X_test_seq, test_units = build_last_window_sequences(test_df, feature_cols, sequence_length=seq_len)
    preds = dl_predict(model, X_test_seq)

    pred_df = pd.DataFrame({"unit_number": test_units, "RUL_pred": preds})
    merged = pred_df.merge(rul_test_gt, on="unit_number", how="left")

    metrics = regression_metrics(merged["RUL"], merged["RUL_pred"])
    _log(f"  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

    plot_rul_scatter(
        merged["RUL"],
        merged["RUL_pred"],
        title="Regression RUL (LSTM) : predit vs reel (test FD001)",
        save_path=FIGURES_DIR / "regression_lstm_scatter.png",
    )

    return {
        "sequence_length": seq_len,
        "n_training_sequences": int(X_seq.shape[0]),
        "epochs": 15,
        "training_time_seconds": round(elapsed, 1),
        "final_train_loss": history.train_loss[-1],
        "final_val_loss": history.val_loss[-1],
        "test_metrics": metrics,
    }


def generate_overview_figure(data: dict) -> None:
    _log("Figure : courbes de degradation des capteurs...")
    plot_sensor_degradation(
        data["train_raw"],
        sensor_cols=["sensor_2", "sensor_7", "sensor_11", "sensor_15"],
        save_path=FIGURES_DIR / "sensor_degradation.png",
        n_units=6,
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare_data()
    generate_overview_figure(data)

    results = {}
    results["supervised"] = run_supervised(data)
    results["unsupervised"] = run_unsupervised(data)
    results["deep_learning"] = run_deep_learning(data)
    results["dataset_info"] = {
        "n_train_units": int(data["train_raw"]["unit_number"].nunique()),
        "n_test_units": int(data["test_raw"]["unit_number"].nunique()),
        "n_train_rows": int(data["train_raw"].shape[0]),
        "n_test_rows": int(data["test_raw"].shape[0]),
        "n_features": len(data["feature_cols"]),
        "rul_cap": RUL_CAP,
        "maintenance_soon_threshold": MAINTENANCE_SOON_THRESHOLD,
    }

    results_path = REPORTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    _log(f"Resultats exportes vers {results_path}")
    _log("Pipeline termine avec succes.")


if __name__ == "__main__":
    main()
