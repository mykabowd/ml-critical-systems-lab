# ml-critical-systems-lab

Projet de **data science appliquee a la maintenance predictive de systemes
critiques**, base sur le jeu de donnees public **NASA C-MAPSS** (Turbofan
Engine Degradation Simulation). Trois familles de modeles de machine
learning sont entrainees et evaluees avec de vraies metriques :

- **Machine learning supervise** (scikit-learn) : regression du RUL
  (Remaining Useful Life) et classification binaire "maintenance necessaire
  bientot"
- **Apprentissage non supervise / detection d'anomalies** (scikit-learn) :
  Isolation Forest et erreur de reconstruction PCA sur les lectures capteurs
- **Deep learning + series temporelles** (PyTorch, CPU) : un LSTM leger
  operant sur des fenetres glissantes de mesures capteurs
- **Visualisation de donnees** (matplotlib / seaborn)
- **Rapport technique** rigoureux et honnete (`reports/technical_report.md`)
- **Tests automatises** (pytest)

## Pourquoi ce projet

Ce projet a ete construit pour combler, de facon **honnete et concrete**,
l'ecart entre un profil d'ingenieur logiciel (Flutter, Java, backend,
Docker/Kubernetes) et les exigences de data science classique d'un poste
vise chez Thales : **"Applied AI Research Scientist – Data Scientist"**
(reference **R0335173**, Quebec). Ce poste demande explicitement :

- une maitrise de Python pour le traitement/nettoyage/analyse de donnees ;
- une comprehension solide du ML supervise, non supervise, du deep
  learning, de la detection d'anomalies et de l'analyse de series
  temporelles ;
- une experience d'entrainement/evaluation de modeles IA/ML ;
- une experience avec des outils de visualisation de donnees ;
- une base solide en mathematiques, statistiques et probabilites ;
- une capacite demontree a mener un projet de recherche appliquee de bout
  en bout, avec peu d'encadrement ;
- une capacite a communiquer un travail scientifique via des **rapports
  techniques**.

Plutot que de pretendre a une experience ou a des publications
inexistantes, ce projet construit un vrai petit projet de recherche
appliquee — donnees reelles telechargees, code qui tourne, metriques
reellement calculees (jamais inventees) — de sorte que chaque ligne de CV
qui en decoule soit strictement vraie et verifiable.

Un projet sœur, [`genai-mcp-assistant`](../genai-mcp-assistant), comble une
autre partie de l'ecart (RAG, LLM, agents LangGraph, serveur MCP,
Docker/Kubernetes/Helm) pour des postes GenAI. Ce projet-ci est **independant** et se
concentre exclusivement sur le ML classique, les statistiques, la
visualisation et la redaction d'un rapport technique — il ne reintroduit
volontairement pas de conteneurisation/orchestration (deja couverte par le
projet sœur).

## Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.10+ |
| Traitement de donnees | pandas, numpy |
| ML supervise / non supervise | scikit-learn (RandomForest, LogisticRegression, LinearRegression, IsolationForest, PCA) |
| Deep learning | PyTorch (CPU) — LSTM |
| Visualisation | matplotlib, seaborn |
| Tests | pytest |
| Packaging | setuptools (src-layout), `pyproject.toml` |

## Demarrage rapide

```bash
make venv install     # cree le venv (.venv/) et installe les dependances + le package en mode editable
make download-data    # telecharge et extrait le jeu de donnees NASA C-MAPSS (sous-ensemble FD001) dans data/raw/
make pipeline         # execute le pipeline complet : pretraitement -> 3 familles de modeles -> figures -> reports/results.json
make test             # execute les tests automatises (pytest)
```

Le pipeline complet (`make pipeline`) prend environ 1 a 2 minutes sur un
ordinateur portable standard (CPU uniquement, pas de GPU requis), dont
~35 secondes pour l'entrainement du LSTM.

## Structure du projet

```
ml-critical-systems-lab/
├── data/
│   └── raw/                        # donnees NASA C-MAPSS brutes (non versionnees, cf. .gitignore)
├── scripts/
│   └── download_data.py            # telecharge + extrait le zip NASA (FD001)
├── src/ml_critical_systems_lab/
│   ├── config.py                    # chemins et constantes centralisees (cap RUL, seuils, capteurs retenus)
│   ├── data.py                      # chargement train/test/RUL en DataFrame pandas
│   ├── features.py                  # calcul du RUL, plafonnement, stats glissantes, normalisation
│   ├── metrics.py                    # RMSE/MAE/R2, precision/recall/F1, AUC anomalies, correlation Spearman
│   ├── supervised.py                 # regression RUL (LinearRegression, RandomForestRegressor) + classification (LogisticRegression, RandomForestClassifier)
│   ├── unsupervised.py               # detection d'anomalies (IsolationForest, PCA)
│   ├── deep_learning.py              # LSTM PyTorch + construction de sequences temporelles
│   ├── visualize.py                   # generation des figures (matplotlib/seaborn)
│   └── pipeline.py                    # orchestrateur de bout en bout
├── tests/                            # tests pytest (donnees synthetiques, deterministes)
├── reports/
│   ├── technical_report.md           # rapport technique complet (methodologie + resultats reels)
│   ├── results.json                   # metriques exportees automatiquement par le pipeline
│   └── figures/                       # figures generees (PNG)
├── pyproject.toml
├── requirements.txt
├── Makefile
└── .gitignore
```

## Resultats cles (extraits du rapport technique)

Toutes les valeurs ci-dessous sont produites par une execution reelle du
pipeline sur le sous-ensemble **FD001** de C-MAPSS (100 moteurs
d'entrainement, 100 moteurs de test) — voir
[`reports/technical_report.md`](reports/technical_report.md) pour le detail
complet et les figures.

| Famille | Meilleur modele | Metrique | Valeur |
|---|---|---|---|
| Supervise (regression RUL) | Random Forest Regressor | RMSE | 18.74 cycles |
| Supervise (classification "maintenance bientot") | Random Forest Classifier | F1 | 0.870 |
| Non supervise (detection d'anomalies) | Isolation Forest | AUC | 0.892 |
| Deep learning (LSTM, series temporelles) | LSTM (1 couche, CPU, 15 epochs) | RMSE | 18.98 cycles |

## Ce que ce projet demontre (et ce qu'il ne demontre pas)

**Demontre** : traitement et nettoyage de donnees en Python (pandas),
feature engineering sur series temporelles multivariees, machine learning
supervise (regression et classification, scikit-learn), apprentissage non
supervise et detection d'anomalies (Isolation Forest, PCA), deep learning
sur CPU avec PyTorch (LSTM), analyse de series temporelles, visualisation de
donnees (matplotlib/seaborn), calcul et interpretation honnete de vraies
metriques (RMSE, precision/recall/F1, AUC, correlation), redaction d'un
rapport technique structure, tests automatises, developpement realise sous
Linux/macOS en ligne de commande.

**Ne demontre PAS** (a ne pas sur-vendre sur un CV) : traitement Big
Data/Spark (le jeu de donnees tient en memoire, ~12 Mo), deploiement en
production (pas d'API de service, pas de monitoring de derive), publication
scientifique evaluee par des pairs (ce rapport est un document technique
interne, pas un article revise), recherche d'hyperparametres exhaustive
(valeurs raisonnables issues de la litterature, pas de grid search
systematique a grande echelle), traitement de donnees de capteurs reelles
(C-MAPSS est une simulation NASA).

## Limites connues / pistes d'amelioration

Voir la section 6 de [`reports/technical_report.md`](reports/technical_report.md)
pour le detail complet (dataset simule, sous-ensemble le plus simple choisi
volontairement, modele de deep learning volontairement reduit, PCA peu
performante sur ce jeu de donnees, pas de recherche d'hyperparametres
exhaustive, pas de deploiement).
