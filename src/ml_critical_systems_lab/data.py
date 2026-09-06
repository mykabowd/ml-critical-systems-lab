"""Chargement des fichiers bruts NASA C-MAPSS (sous-ensemble FD001) en
DataFrame pandas, avec noms de colonnes explicites.

Format des fichiers source : texte, colonnes separees par des espaces,
26 colonnes (unit_number, time_in_cycles, 3 reglages operationnels,
21 mesures de capteurs), avec un espace final en fin de ligne.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_critical_systems_lab.config import (
    COLUMN_NAMES,
    RUL_FILE,
    TEST_FILE,
    TRAIN_FILE,
)


def _load_cmapss_txt(path: Path) -> pd.DataFrame:
    """Charge un fichier train_FD00X.txt ou test_FD00X.txt en DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Avez-vous execute `python scripts/download_data.py` ?"
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)
    return df


def load_train(path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Charge le jeu d'entrainement FD001 (trajectoires completes jusqu'a panne)."""
    return _load_cmapss_txt(path)


def load_test(path: Path = TEST_FILE) -> pd.DataFrame:
    """Charge le jeu de test FD001 (trajectoires tronquees avant la panne)."""
    return _load_cmapss_txt(path)


def load_rul(path: Path = RUL_FILE) -> pd.DataFrame:
    """Charge le vecteur de veritee terrain RUL pour chaque unite du jeu de test.

    Le fichier RUL_FD001.txt contient une valeur par ligne, dans le meme
    ordre que les unites du jeu de test (unit_number = index de ligne + 1).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Avez-vous execute `python scripts/download_data.py` ?"
        )

    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"], engine="python")
    rul.insert(0, "unit_number", range(1, len(rul) + 1))
    return rul
