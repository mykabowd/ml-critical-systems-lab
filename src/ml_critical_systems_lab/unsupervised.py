"""Detection d'anomalies non supervisee sur les lectures capteurs :
Isolation Forest et erreur de reconstruction PCA.

Aucun de ces modeles n'utilise le label RUL pendant l'entrainement : ils
apprennent uniquement a partir des mesures de capteurs, ce qui correspond a
un scenario realiste ou l'on cherche a detecter des etats degrades sans
etiquette de panne disponible. Le RUL n'est utilise qu'a posteriori, pour
*evaluer* si le score d'anomalie produit est effectivement plus eleve pres
de la panne (cf. `metrics.anomaly_detection_auc`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from ml_critical_systems_lab.config import RANDOM_STATE


def fit_isolation_forest(
    X: pd.DataFrame, n_estimators: int = 200, contamination: str | float = "auto"
) -> IsolationForest:
    """Ajuste un Isolation Forest sur des lectures capteurs normalisees."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def isolation_forest_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Score d'anomalie : plus la valeur est elevee, plus l'observation est
    consideree comme anormale (convention inversee par rapport a
    `score_samples` de scikit-learn, ou une valeur elevee = normal).
    """
    return -model.score_samples(X)


def fit_pca(X: pd.DataFrame, n_components: int) -> PCA:
    """Ajuste une PCA sur des lectures capteurs normalisees."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(X)
    return pca


def pca_reconstruction_error(pca: PCA, X: pd.DataFrame) -> np.ndarray:
    """Erreur de reconstruction (somme des carres des residus) apres
    projection sur les `n_components` premieres composantes puis
    reconstruction. Une erreur elevee indique un ecart par rapport aux
    modes de variation "normaux" appris par la PCA, donc un etat potentiellement
    degrade.
    """
    X_arr = np.asarray(X)
    X_projected = pca.transform(X_arr)
    X_reconstructed = pca.inverse_transform(X_projected)
    return np.sum((X_arr - X_reconstructed) ** 2, axis=1)
