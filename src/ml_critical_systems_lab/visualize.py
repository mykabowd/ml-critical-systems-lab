"""Visualisations (matplotlib/seaborn) sauvegardees dans reports/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif : execution en pipeline/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_sensor_degradation(
    df: pd.DataFrame,
    sensor_cols: list[str],
    save_path: Path,
    n_units: int = 6,
    seed: int = 42,
) -> None:
    """Courbes de degradation de quelques capteurs au fil des cycles, pour
    plusieurs unites superposees (jeu d'entrainement, trajectoires completes
    jusqu'a la panne).
    """
    rng = np.random.RandomState(seed)
    units = df["unit_number"].unique()
    chosen_units = [int(u) for u in rng.choice(units, size=min(n_units, len(units)), replace=False)]

    n_sensors = len(sensor_cols)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(9, 2.6 * n_sensors), sharex=False)
    if n_sensors == 1:
        axes = [axes]

    for ax, sensor in zip(axes, sensor_cols):
        for unit in chosen_units:
            unit_df = df[df["unit_number"] == unit]
            ax.plot(unit_df["time_in_cycles"], unit_df[sensor], alpha=0.8, linewidth=1)
        ax.set_ylabel(sensor)
        ax.set_title(f"{sensor} en fonction du cycle (unites: {list(chosen_units)})", fontsize=9)

    axes[-1].set_xlabel("cycle")
    fig.suptitle("Degradation des capteurs au fil des cycles (jeu d'entrainement FD001)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rul_scatter(
    y_true,
    y_pred,
    title: str,
    save_path: Path,
) -> None:
    """Scatter RUL predit vs RUL reel, avec la droite y=x en reference."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="prediction parfaite (y=x)")
    ax.set_xlabel("RUL reel")
    ax.set_ylabel("RUL predit")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Matrice de confusion sous forme de heatmap annotee."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Classe reelle")
    ax.set_xlabel("Classe predite")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_anomaly_score_distribution(
    scores_normal,
    scores_near_failure,
    title: str,
    save_path: Path,
) -> None:
    """Distribution du score d'anomalie : etats normaux vs proches de la panne."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(scores_normal, color="steelblue", label="normal (RUL eleve)", stat="density", kde=True, alpha=0.5, ax=ax)
    sns.histplot(scores_near_failure, color="crimson", label="proche de la panne (RUL faible)", stat="density", kde=True, alpha=0.5, ax=ax)
    ax.set_xlabel("score d'anomalie")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curve(train_loss: list[float], val_loss: list[float], save_path: Path) -> None:
    """Courbe de perte d'entrainement/validation (modele PyTorch)."""
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="train loss (MSE)", marker="o", markersize=3)
    ax.plot(epochs, val_loss, label="val loss (MSE)", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Courbe d'entrainement du LSTM (RUL)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
