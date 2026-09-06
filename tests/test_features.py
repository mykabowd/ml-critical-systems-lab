"""Tests du feature engineering (features.py) : calcul du RUL, plafonnement,
statistiques glissantes, normalisation. Donnees synthetiques, rapides et
deterministes -- aucun entrainement de modele ici.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_critical_systems_lab.features import (
    add_maintenance_soon_label,
    add_rolling_features,
    compute_rul,
    get_last_cycle_per_unit,
    scale_features,
)


def _make_synthetic_units() -> pd.DataFrame:
    # Unite 1 : 5 cycles (1..5). Unite 2 : 3 cycles (1..3).
    rows = []
    for unit, n_cycles in [(1, 5), (2, 3)]:
        for cycle in range(1, n_cycles + 1):
            rows.append({"unit_number": unit, "time_in_cycles": cycle, "sensor_x": float(cycle)})
    return pd.DataFrame(rows)


def test_compute_rul_uncapped():
    df = _make_synthetic_units()
    result = compute_rul(df, cap=None)

    unit1 = result[result["unit_number"] == 1].sort_values("time_in_cycles")
    assert unit1["RUL"].tolist() == [4, 3, 2, 1, 0]

    unit2 = result[result["unit_number"] == 2].sort_values("time_in_cycles")
    assert unit2["RUL"].tolist() == [2, 1, 0]


def test_compute_rul_capped():
    df = _make_synthetic_units()
    result = compute_rul(df, cap=2)

    unit1 = result[result["unit_number"] == 1].sort_values("time_in_cycles")
    # RUL non plafonne = [4, 3, 2, 1, 0] -> plafonne a 2 = [2, 2, 2, 1, 0]
    assert unit1["RUL"].tolist() == [2, 2, 2, 1, 0]


def test_add_maintenance_soon_label_threshold():
    df = pd.DataFrame({"RUL": [50, 29, 30, 0, 125]})
    result = add_maintenance_soon_label(df, threshold=30)
    assert result["maintenance_soon"].tolist() == [0, 1, 0, 1, 0]


def test_add_maintenance_soon_label_requires_rul_column():
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    with pytest.raises(ValueError):
        add_maintenance_soon_label(df)


def test_add_rolling_features_matches_manual_calculation():
    df = _make_synthetic_units()
    result = add_rolling_features(df, sensor_cols=["sensor_x"], window=2)

    unit1 = result[result["unit_number"] == 1].sort_values("time_in_cycles")
    # sensor_x pour unite 1 = [1, 2, 3, 4, 5], fenetre=2, min_periods=1 :
    # roll_mean = [1, 1.5, 2.5, 3.5, 4.5]
    expected_mean = [1.0, 1.5, 2.5, 3.5, 4.5]
    np.testing.assert_allclose(unit1["sensor_x_roll_mean"].tolist(), expected_mean)

    # Les stats glissantes ne doivent jamais melanger deux unites : le premier
    # point de l'unite 2 doit etre egal a sa propre valeur (pas influence par
    # la fin de l'unite 1).
    unit2 = result[result["unit_number"] == 2].sort_values("time_in_cycles")
    assert unit2["sensor_x_roll_mean"].iloc[0] == 1.0


def test_get_last_cycle_per_unit():
    df = _make_synthetic_units()
    last = get_last_cycle_per_unit(df)

    assert last.shape[0] == 2
    assert set(last["unit_number"]) == {1, 2}
    unit1_last = last[last["unit_number"] == 1].iloc[0]
    assert unit1_last["time_in_cycles"] == 5
    unit2_last = last[last["unit_number"] == 2].iloc[0]
    assert unit2_last["time_in_cycles"] == 3


def test_scale_features_fits_only_on_train():
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test_df = pd.DataFrame({"f1": [100.0, 200.0]})  # tres different du train

    scaled = scale_features(train_df, test_df, feature_cols=["f1"])

    # Le train normalise doit avoir une moyenne ~0 et un ecart-type ~1.
    assert scaled.train["f1"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled.train["f1"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    # Le test est transforme avec le scaler du train (pas re-ajuste) : des
    # valeurs tres eloignees du train doivent donner des scores tres eleves.
    assert scaled.test["f1"].iloc[0] > 10
