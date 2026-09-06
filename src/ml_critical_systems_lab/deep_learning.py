"""Modele de deep learning (PyTorch, CPU) pour predire le RUL a partir de
fenetres temporelles de mesures capteurs : un petit LSTM, volontairement peu
profond (1 couche recurrente, hidden_size reduit) pour s'entrainer en
quelques minutes sur CPU sans GPU.

Ce module demontre conjointement "deep learning" et "series temporelles" :
le modele consomme des sequences de longueur fixe (fenetre glissante sur les
cycles d'un moteur) plutot qu'une observation isolee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_critical_systems_lab.config import RANDOM_STATE

DEFAULT_SEQUENCE_LENGTH = 30


class CmapssSequenceDataset(Dataset):
    """Dataset PyTorch : sequences [sequence_length, n_features] -> RUL cible."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


def _pad_or_truncate(unit_array: np.ndarray, sequence_length: int) -> np.ndarray:
    """Si l'unite a moins de cycles que `sequence_length`, on repete la
    premiere ligne au debut (padding "en amont") ; sinon on ne fait rien ici
    (la fenetre glissante s'occupe de la troncature).
    """
    n = unit_array.shape[0]
    if n >= sequence_length:
        return unit_array
    pad_len = sequence_length - n
    first_row = unit_array[0:1, :]
    padding = np.repeat(first_row, pad_len, axis=0)
    return np.concatenate([padding, unit_array], axis=0)


def build_training_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    target_col: str = "RUL",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit des sequences glissantes pour l'entrainement : pour chaque
    unite, toutes les fenetres de longueur `sequence_length` possibles, avec
    comme cible le RUL au dernier cycle de la fenetre.

    Returns:
        (sequences, targets, unit_numbers) ou sequences a la forme
        [n_windows, sequence_length, n_features].
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    unit_numbers: list[int] = []

    for unit, group in df.groupby("unit_number", sort=True):
        group = group.sort_values("time_in_cycles")
        features = group[feature_cols].to_numpy()
        target = group[target_col].to_numpy()

        features = _pad_or_truncate(features, sequence_length)
        if features.shape[0] != group.shape[0]:
            # Le padding a rallonge `features` mais pas `target` : on aligne
            # `target` en repetant sa premiere valeur autant de fois.
            pad_len = features.shape[0] - group.shape[0]
            target = np.concatenate([np.repeat(target[0], pad_len), target])

        n_windows = features.shape[0] - sequence_length + 1
        for start in range(n_windows):
            end = start + sequence_length
            sequences.append(features[start:end])
            targets.append(target[end - 1])
            unit_numbers.append(unit)

    return np.array(sequences), np.array(targets, dtype=np.float32), np.array(unit_numbers)


def build_last_window_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit, pour chaque unite, une seule sequence correspondant aux
    `sequence_length` derniers cycles observes (avec padding si l'unite a
    moins de cycles). Utilise pour l'evaluation sur le jeu de test C-MAPSS,
    ou seul le dernier cycle observe de chaque unite est compare a la
    veritee terrain (RUL_FD001.txt).
    """
    sequences: list[np.ndarray] = []
    unit_numbers: list[int] = []

    for unit, group in df.groupby("unit_number", sort=True):
        group = group.sort_values("time_in_cycles")
        features = group[feature_cols].to_numpy()
        features = _pad_or_truncate(features, sequence_length)
        sequences.append(features[-sequence_length:])
        unit_numbers.append(unit)

    return np.array(sequences), np.array(unit_numbers)


class LSTMRULRegressor(nn.Module):
    """Petit LSTM (1 couche, hidden_size reduit) + tete de regression lineaire."""

    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, hidden_size]
        out = self.head(last_hidden)
        return out.squeeze(-1)


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)


def train_lstm(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> TrainingHistory:
    """Boucle d'entrainement simple (MSE loss, Adam), sur CPU."""
    torch.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = TrainingHistory()

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                epoch_val_losses.append(loss.item())

        train_loss = float(np.mean(epoch_train_losses))
        val_loss = float(np.mean(epoch_val_losses))
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        print(f"  epoch {epoch + 1}/{epochs} - train_loss={train_loss:.3f} - val_loss={val_loss:.3f}")

    return history


def predict(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
    """Predictions du modele sur un batch de sequences [n, seq_len, n_features]."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(sequences, dtype=torch.float32)
        preds = model(X)
    return preds.numpy()
