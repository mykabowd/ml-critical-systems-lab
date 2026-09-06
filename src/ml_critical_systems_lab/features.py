"""Feature engineering pour C-MAPSS : calcul du label RUL, statistiques
glissantes sur les capteurs, et normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_critical_systems_lab.config import (
    MAINTENANCE_SOON_THRESHOLD,
    ROLLING_SENSOR_COLS,
    ROLLING_WINDOW,
    RUL_CAP,
)


def compute_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Calcule le label RUL (Remaining Useful Life) par unite.

    Pour chaque unite (moteur), RUL au cycle t = (dernier cycle observe pour
    cette unite) - t. C'est la definition standard pour le jeu d'entrainement
    C-MAPSS, ou chaque trajectoire va jusqu'a la panne.

    Convention de la litterature (Heimes 2008, Saxena & Goebel 2008) : le RUL
    est plafonne (`cap`, 125 cycles par defaut) car la degradation n'est en
    pratique significative que dans les derniers cycles avant la panne ; sans
    plafond, le modele est penalise pour ne pas predire une degradation
    lineaire tres en amont, alors qu'aucun signal de degradation n'est encore
    present dans les capteurs a ce stade.

    Args:
        df: DataFrame avec colonnes `unit_number`, `time_in_cycles`.
        cap: valeur maximale du RUL (None pour ne pas plafonner).

    Returns:
        Copie de `df` avec une colonne `RUL` ajoutee.
    """
    df = df.copy()
    max_cycle_per_unit = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycle_per_unit - df["time_in_cycles"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_maintenance_soon_label(
    df: pd.DataFrame, threshold: int = MAINTENANCE_SOON_THRESHOLD
) -> pd.DataFrame:
    """Ajoute un label binaire `maintenance_soon` = 1 si RUL < threshold.

    Necessite que la colonne `RUL` existe deja (cf. `compute_rul`).
    """
    if "RUL" not in df.columns:
        raise ValueError("La colonne 'RUL' doit exister avant d'appeler cette fonction.")
    df = df.copy()
    df["maintenance_soon"] = (df["RUL"] < threshold).astype(int)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str] = ROLLING_SENSOR_COLS,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Ajoute des statistiques glissantes (moyenne, ecart-type) par unite.

    Le calcul est fait separement pour chaque unite (`unit_number`) afin de
    ne jamais melanger les cycles de deux moteurs differents. Les premieres
    lignes de chaque unite (fenetre incomplete) sont remplies via une fenetre
    expansive (min_periods=1) pour eviter des NaN qui reduiraient le jeu de
    donnees disponible.

    Args:
        df: DataFrame trie par `unit_number`, `time_in_cycles`.
        sensor_cols: colonnes de capteurs sur lesquelles calculer les stats.
        window: taille de la fenetre glissante (nombre de cycles).

    Returns:
        Copie de `df` avec, pour chaque capteur `sensor_X`, deux colonnes
        ajoutees : `sensor_X_roll_mean` et `sensor_X_roll_std`.
    """
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
    grouped = df.groupby("unit_number", sort=False)

    new_cols = {}
    for col in sensor_cols:
        rolling = grouped[col].rolling(window=window, min_periods=1)
        new_cols[f"{col}_roll_mean"] = rolling.mean().reset_index(level=0, drop=True)
        new_cols[f"{col}_roll_std"] = rolling.std().reset_index(level=0, drop=True).fillna(0.0)

    result = df.copy()
    for name, series in new_cols.items():
        result[name] = series.values
    return result


@dataclass
class ScaledData:
    """Conteneur pour les donnees normalisees et le scaler associe."""

    train: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler
    feature_cols: list[str]


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> ScaledData:
    """Normalise les colonnes `feature_cols` avec un StandardScaler.

    Le scaler est ajuste (`fit`) uniquement sur le jeu d'entrainement, puis
    applique (`transform`) au train et au test, afin d'eviter toute fuite
    d'information du test vers le train (data leakage).
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return ScaledData(
        train=train_scaled, test=test_scaled, scaler=scaler, feature_cols=feature_cols
    )


def get_last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement la derniere ligne (dernier cycle observe) de chaque
    unite. Utilise pour l'evaluation sur le jeu de test C-MAPSS, ou seule la
    prediction au dernier cycle observe est comparee au RUL_FD001.txt fourni.
    """
    return (
        df.sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def build_feature_columns(sensor_cols: list[str] = ROLLING_SENSOR_COLS) -> list[str]:
    """Liste des colonnes de features utilisees par les modeles supervises :
    reglages operationnels + capteurs bruts + statistiques glissantes.
    """
    from ml_critical_systems_lab.config import OP_SETTING_COLS, SENSOR_COLS

    rolling_cols = []
    for col in sensor_cols:
        rolling_cols.append(f"{col}_roll_mean")
        rolling_cols.append(f"{col}_roll_std")

    return list(OP_SETTING_COLS) + list(SENSOR_COLS) + rolling_cols
