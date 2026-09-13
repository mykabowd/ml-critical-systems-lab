🇫🇷 [Version française](README.fr.md)

# ml-critical-systems-lab

Applied **data science project for predictive maintenance of critical
systems**, based on the public **NASA C-MAPSS** dataset (Turbofan Engine
Degradation Simulation). Three families of machine learning models are
trained and evaluated with real metrics:

- **Supervised machine learning** (scikit-learn): RUL (Remaining Useful
  Life) regression and binary classification for "maintenance needed
  soon"
- **Unsupervised learning / anomaly detection** (scikit-learn): Isolation
  Forest and PCA reconstruction error on sensor readings
- **Deep learning + time series** (PyTorch, CPU): a lightweight LSTM
  operating on sliding windows of sensor measurements
- **Data visualization** (matplotlib / seaborn)
- **Rigorous, honest technical report** (`reports/technical_report.md`)
- **Automated tests** (pytest)

## Why this project

This project was built to close, in an **honest and concrete** way, the
gap between a software engineering profile (Flutter, Java, backend,
Docker/Kubernetes) and the classic data science skill set expected for
**Data Scientist / AI Research Scientist**-type roles. Roles like these
typically require:

- strong Python skills for data processing/cleaning/analysis;
- a solid understanding of supervised and unsupervised ML, deep learning,
  anomaly detection, and time series analysis;
- experience training and evaluating AI/ML models;
- experience with data visualization tools;
- a solid foundation in mathematics, statistics, and probability;
- a demonstrated ability to run an applied research project end-to-end
  with limited supervision;
- the ability to communicate scientific work through **technical
  reports**.

Rather than claiming nonexistent experience or publications, this project
builds a real, small applied research project — real downloaded data,
code that actually runs, metrics that are genuinely computed (never made
up) — so that every line of the resulting résumé is strictly true and
verifiable.

A sister project, [`genai-mcp-assistant`](../genai-mcp-assistant), covers
another part of the gap (RAG, LLMs, LangGraph agents, MCP server,
Docker/Kubernetes/Helm) for GenAI-oriented roles. This project is
**independent** and focuses exclusively on classic ML, statistics,
visualization, and technical report writing — it deliberately does not
reintroduce containerization/orchestration (already covered by the sister
project).

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, numpy |
| Supervised / unsupervised ML | scikit-learn (RandomForest, LogisticRegression, LinearRegression, IsolationForest, PCA) |
| Deep learning | PyTorch (CPU) — LSTM |
| Visualization | matplotlib, seaborn |
| Tests | pytest |
| Packaging | setuptools (src-layout), `pyproject.toml` |

## Quickstart

```bash
make venv install     # creates the venv (.venv/) and installs dependencies + the package in editable mode
make download-data    # downloads and extracts the NASA C-MAPSS dataset (FD001 subset) into data/raw/
make pipeline         # runs the full pipeline: preprocessing -> 3 model families -> figures -> reports/results.json
make test             # runs the automated tests (pytest)
```

The full pipeline (`make pipeline`) takes about 1 to 2 minutes on a
standard laptop (CPU only, no GPU required), including ~35 seconds for
LSTM training.

## Project structure

```
ml-critical-systems-lab/
├── data/
│   └── raw/                        # raw NASA C-MAPSS data (not versioned, see .gitignore)
├── scripts/
│   └── download_data.py            # downloads + extracts the NASA zip (FD001)
├── src/ml_critical_systems_lab/
│   ├── config.py                    # centralized paths and constants (RUL cap, thresholds, selected sensors)
│   ├── data.py                      # loads train/test/RUL into pandas DataFrames
│   ├── features.py                  # RUL computation, capping, rolling stats, normalization
│   ├── metrics.py                    # RMSE/MAE/R2, precision/recall/F1, anomaly AUC, Spearman correlation
│   ├── supervised.py                 # RUL regression (LinearRegression, RandomForestRegressor) + classification (LogisticRegression, RandomForestClassifier)
│   ├── unsupervised.py               # anomaly detection (IsolationForest, PCA)
│   ├── deep_learning.py              # PyTorch LSTM + time series sequence construction
│   ├── visualize.py                   # figure generation (matplotlib/seaborn)
│   └── pipeline.py                    # end-to-end orchestrator
├── tests/                            # pytest tests (synthetic, deterministic data)
├── reports/
│   ├── technical_report.md           # full technical report (methodology + real results)
│   ├── results.json                   # metrics automatically exported by the pipeline
│   └── figures/                       # generated figures (PNG)
├── pyproject.toml
├── requirements.txt
├── Makefile
└── .gitignore
```

## Key results (excerpted from the technical report)

All values below come from an actual run of the pipeline on the
**FD001** subset of C-MAPSS (100 training engines, 100 test engines) —
see [`reports/technical_report.md`](reports/technical_report.md) for full
details and figures.

| Family | Best model | Metric | Value |
|---|---|---|---|
| Supervised (RUL regression) | Random Forest Regressor | RMSE | 18.74 cycles |
| Supervised ("maintenance soon" classification) | Random Forest Classifier | F1 | 0.870 |
| Unsupervised (anomaly detection) | Isolation Forest | AUC | 0.892 |
| Deep learning (LSTM, time series) | LSTM (1 layer, CPU, 15 epochs) | RMSE | 18.98 cycles |

## What this project demonstrates (and what it does not)

**Demonstrates**: data processing and cleaning in Python (pandas),
feature engineering on multivariate time series, supervised machine
learning (regression and classification, scikit-learn), unsupervised
learning and anomaly detection (Isolation Forest, PCA), CPU-based deep
learning with PyTorch (LSTM), time series analysis, data visualization
(matplotlib/seaborn), honest computation and interpretation of real
metrics (RMSE, precision/recall/F1, AUC, correlation), writing a
structured technical report, automated testing, development carried out
on Linux/macOS from the command line.

**Does NOT demonstrate** (should not be oversold on a résumé): Big
Data/Spark-scale processing (the dataset fits in memory, ~12 MB),
production deployment (no serving API, no drift monitoring), peer-reviewed
scientific publication (this report is an internal technical document,
not a reviewed paper), exhaustive hyperparameter search (reasonable
values from the literature, no large-scale systematic grid search),
processing of real-world sensor data (C-MAPSS is a NASA simulation).

## Known limitations / possible improvements

See section 6 of [`reports/technical_report.md`](reports/technical_report.md)
for the full detail (simulated dataset, deliberately chosen simplest
subset, deliberately reduced deep learning model, PCA underperforming on
this dataset, no exhaustive hyperparameter search, no deployment).
