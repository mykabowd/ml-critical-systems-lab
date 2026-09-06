"""Tests du chargement des donnees brutes C-MAPSS (data.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_critical_systems_lab.data import load_rul, load_test, load_train


def _write_cmapss_txt(path, rows: list[list[float]]) -> None:
    """Ecrit un fichier au format C-MAPSS (espaces, espace final par ligne)."""
    lines = []
    for row in rows:
        lines.append(" ".join(str(v) for v in row) + " ")
    path.write_text("\n".join(lines) + "\n")


def _fake_row(unit: int, cycle: int) -> list[float]:
    # unit, cycle, 3 op_settings, 21 capteurs (valeurs arbitraires simples).
    return [unit, cycle, 0.001, 0.0002, 100.0] + [100.0 + i for i in range(21)]


def test_load_train_shape_and_columns(tmp_path):
    rows = [_fake_row(1, 1), _fake_row(1, 2), _fake_row(2, 1)]
    path = tmp_path / "train_FD001.txt"
    _write_cmapss_txt(path, rows)

    df = load_train(path)

    assert df.shape == (3, 26)
    assert list(df.columns[:5]) == [
        "unit_number",
        "time_in_cycles",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ]
    assert df["unit_number"].tolist() == [1, 1, 2]
    assert df["time_in_cycles"].dtype.kind == "i"


def test_load_test_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        load_test(missing_path)


def test_load_rul_assigns_unit_numbers(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112 \n98 \n7 \n")

    df = load_rul(path)

    assert isinstance(df, pd.DataFrame)
    assert df["unit_number"].tolist() == [1, 2, 3]
    assert df["RUL"].tolist() == [112, 98, 7]
