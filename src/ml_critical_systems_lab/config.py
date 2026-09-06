"""Configuration centralisee : chemins de fichiers et constantes du projet."""

from __future__ import annotations

from pathlib import Path

# Racine du projet (2 niveaux au-dessus de ce fichier : src/ml_critical_systems_lab/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = PROJECT_ROOT / "models"

# Sous-ensemble C-MAPSS utilise : FD001 (une seule condition operationnelle,
# un seul mode de panne -- le plus simple du jeu de donnees NASA C-MAPSS).
SUBSET = "FD001"

TRAIN_FILE = RAW_DATA_DIR / f"train_{SUBSET}.txt"
TEST_FILE = RAW_DATA_DIR / f"test_{SUBSET}.txt"
RUL_FILE = RAW_DATA_DIR / f"RUL_{SUBSET}.txt"

# Noms de colonnes explicites (26 colonnes, cf. readme.txt du dataset NASA).
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = ["unit_number", "time_in_cycles"] + OP_SETTING_COLS + SENSOR_COLS

# Plafond standard de la litterature C-MAPSS pour le label RUL (evite que le
# modele essaie d'apprendre une degradation lineaire tres en amont de la panne,
# alors qu'en pratique la degradation n'est significative que pres de la fin
# de vie). Reference : Heimes (2008), Saxena & Goebel (2008 PHM Challenge).
RUL_CAP = 125

# Seuil pour la classification binaire "maintenance necessaire bientot".
MAINTENANCE_SOON_THRESHOLD = 30

# Capteurs jugesutiles pour le feature engineering (retires : capteurs
# constants ou quasi-constants sur FD001, identifies par inspection des donnees,
# cf. reports/technical_report.md section Methodologie).
ROLLING_SENSOR_COLS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_20",
    "sensor_21",
]

ROLLING_WINDOW = 5

RANDOM_STATE = 42
