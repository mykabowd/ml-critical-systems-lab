"""Telecharge et extrait le jeu de donnees NASA C-MAPSS (Turbofan Engine
Degradation Simulation Data Set), et place le sous-ensemble FD001 dans
`data/raw/`.

Le zip source contient un dossier avec un zip imbriqu (`CMAPSSData.zip`) qui
contient lui-meme les fichiers texte utiles (train_FD00X.txt, test_FD00X.txt,
RUL_FD00X.txt, readme.txt, ...).

Usage :
    python scripts/download_data.py
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from pathlib import Path

import requests

DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# On ne garde que le sous-ensemble FD001 (une seule condition operationnelle,
# un seul mode de panne) + le readme du dataset complet, pour rester simple.
FILES_TO_KEEP = {
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
    "readme.txt",
    "Damage Propagation Modeling.pdf",
}


def download_zip_bytes(url: str) -> bytes:
    print(f"Telechargement depuis {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    size_mb = len(response.content) / (1024 * 1024)
    print(f"  -> {size_mb:.2f} Mo recus (HTTP {response.status_code})")
    return response.content


def extract_nested_zip(outer_zip_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(outer_zip_bytes)) as outer_zip:
        names = outer_zip.namelist()
        print(f"Zip externe : {len(names)} entrees.")

        # Cherche le zip imbrique CMAPSSData.zip (le nom exact peut varier
        # legerement selon l'archive, ex. dossier parent).
        nested_zip_names = [n for n in names if n.endswith("CMAPSSData.zip")]

        if nested_zip_names:
            nested_name = nested_zip_names[0]
            print(f"Zip imbrique trouve : {nested_name}")
            nested_bytes = outer_zip.read(nested_name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner_zip:
                _extract_matching_files(inner_zip, dest_dir)
        else:
            # Repli : les fichiers .txt sont peut-etre directement dans le
            # zip externe (variante possible de l'archive NASA).
            print("Pas de zip imbrique detecte, recherche directe des .txt ...")
            _extract_matching_files(outer_zip, dest_dir)


def _extract_matching_files(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    extracted = []
    for info in zf.infolist():
        base_name = Path(info.filename).name
        if base_name in FILES_TO_KEEP:
            target_path = dest_dir / base_name
            with zf.open(info) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(base_name)

    if not extracted:
        available = [Path(i.filename).name for i in zf.infolist()]
        raise RuntimeError(
            "Aucun fichier attendu trouve dans l'archive. "
            f"Fichiers disponibles : {available}"
        )

    print(f"Fichiers extraits vers {dest_dir} : {sorted(extracted)}")


def main() -> None:
    required = {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"}
    already_present = {f for f in required if (RAW_DATA_DIR / f).exists()}
    if already_present == required:
        print(f"Donnees deja presentes dans {RAW_DATA_DIR}, rien a faire.")
        return

    try:
        zip_bytes = download_zip_bytes(DATASET_URL)
    except requests.RequestException as exc:
        print(f"ERREUR lors du telechargement : {exc}", file=sys.stderr)
        sys.exit(1)

    extract_nested_zip(zip_bytes, RAW_DATA_DIR)

    missing = {f for f in required if not (RAW_DATA_DIR / f).exists()}
    if missing:
        print(f"ERREUR : fichiers manquants apres extraction : {missing}", file=sys.stderr)
        sys.exit(1)

    print("Telechargement et extraction termines avec succes.")


if __name__ == "__main__":
    main()
