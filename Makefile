.PHONY: venv install download-data pipeline test clean

venv:
	python3 -m venv .venv

install:
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

download-data:
	.venv/bin/python scripts/download_data.py

pipeline:
	.venv/bin/python -m ml_critical_systems_lab.pipeline

test:
	.venv/bin/pytest -v

clean:
	rm -rf reports/figures/*.png models/ data/interim data/processed
