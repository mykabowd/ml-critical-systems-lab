🇬🇧 [English version](README.md)

# ml-critical-systems-lab

Projet de **data science appliquée à la maintenance prédictive de systèmes
critiques**, basé sur le jeu de données public **NASA C-MAPSS** (Turbofan
Engine Degradation Simulation). Trois familles de modèles de machine
learning sont entraînées et évaluées avec de vraies métriques :

- **Machine learning supervisé** (scikit-learn) : régression du RUL
  (Remaining Useful Life) et classification binaire "maintenance nécessaire
  bientôt"
- **Apprentissage non supervisé / détection d'anomalies** (scikit-learn) :
  Isolation Forest et erreur de reconstruction PCA sur les lectures capteurs
- **Deep learning + séries temporelles** (PyTorch, CPU) : un LSTM léger
  opérant sur des fenêtres glissantes de mesures capteurs
- **Visualisation de données** (matplotlib / seaborn)
- **Rapport technique** rigoureux et honnête (`reports/technical_report.md`)
- **Tests automatisés** (pytest)

## Pourquoi ce projet

Ce projet a été construit pour combler, de façon **honnête et concrète**,
l'écart entre un profil d'ingénieur logiciel (Flutter, Java, backend,
Docker/Kubernetes) et les compétences classiques attendues pour des rôles
de type **Data Scientist / AI Research Scientist**. Ce type de poste
demande typiquement :

- une maîtrise de Python pour le traitement/nettoyage/analyse de données ;
- une compréhension solide du ML supervisé, non supervisé, du deep
  learning, de la détection d'anomalies et de l'analyse de séries
  temporelles ;
- une expérience d'entraînement/évaluation de modèles IA/ML ;
- une expérience avec des outils de visualisation de données ;
- une base solide en mathématiques, statistiques et probabilités ;
- une capacité démontrée à mener un projet de recherche appliquée de bout
  en bout, avec peu d'encadrement ;
- une capacité à communiquer un travail scientifique via des **rapports
  techniques**.

Plutôt que de prétendre à une expérience ou à des publications
inexistantes, ce projet construit un vrai petit projet de recherche
appliquée — données réelles téléchargées, code qui tourne, métriques
réellement calculées (jamais inventées) — de sorte que chaque ligne de CV
qui en découle soit strictement vraie et vérifiable.

Un projet sœur, [`genai-mcp-assistant`](../genai-mcp-assistant), comble une
autre partie de l'écart (RAG, LLM, agents LangGraph, serveur MCP,
Docker/Kubernetes/Helm) pour des postes GenAI. Ce projet-ci est
**indépendant** et se concentre exclusivement sur le ML classique, les
statistiques, la visualisation et la rédaction d'un rapport technique — il
ne réintroduit volontairement pas de conteneurisation/orchestration (déjà
couverte par le projet sœur).

## Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.10+ |
| Traitement de données | pandas, numpy |
| ML supervisé / non supervisé | scikit-learn (RandomForest, LogisticRegression, LinearRegression, IsolationForest, PCA) |
| Deep learning | PyTorch (CPU) — LSTM |
| Visualisation | matplotlib, seaborn |
| Tests | pytest |
| Packaging | setuptools (src-layout), `pyproject.toml` |

## Démarrage rapide

```bash
make venv install     # crée le venv (.venv/) et installe les dépendances + le package en mode editable
make download-data    # télécharge et extrait le jeu de données NASA C-MAPSS (sous-ensemble FD001) dans data/raw/
make pipeline         # exécute le pipeline complet : prétraitement -> 3 familles de modèles -> figures -> reports/results.json
make test             # exécute les tests automatisés (pytest)
```

Le pipeline complet (`make pipeline`) prend environ 1 à 2 minutes sur un
ordinateur portable standard (CPU uniquement, pas de GPU requis), dont
~35 secondes pour l'entraînement du LSTM.

## Structure du projet

```
ml-critical-systems-lab/
├── data/
│   └── raw/                        # données NASA C-MAPSS brutes (non versionnées, cf. .gitignore)
├── scripts/
│   └── download_data.py            # télécharge + extrait le zip NASA (FD001)
├── src/ml_critical_systems_lab/
│   ├── config.py                    # chemins et constantes centralisées (cap RUL, seuils, capteurs retenus)
│   ├── data.py                      # chargement train/test/RUL en DataFrame pandas
│   ├── features.py                  # calcul du RUL, plafonnement, stats glissantes, normalisation
│   ├── metrics.py                    # RMSE/MAE/R2, precision/recall/F1, AUC anomalies, corrélation Spearman
│   ├── supervised.py                 # régression RUL (LinearRegression, RandomForestRegressor) + classification (LogisticRegression, RandomForestClassifier)
│   ├── unsupervised.py               # détection d'anomalies (IsolationForest, PCA)
│   ├── deep_learning.py              # LSTM PyTorch + construction de séquences temporelles
│   ├── visualize.py                   # génération des figures (matplotlib/seaborn)
│   └── pipeline.py                    # orchestrateur de bout en bout
├── tests/                            # tests pytest (données synthétiques, déterministes)
├── reports/
│   ├── technical_report.md           # rapport technique complet (méthodologie + résultats réels)
│   ├── results.json                   # métriques exportées automatiquement par le pipeline
│   └── figures/                       # figures générées (PNG)
├── pyproject.toml
├── requirements.txt
├── Makefile
└── .gitignore
```

## Résultats clés (extraits du rapport technique)

Toutes les valeurs ci-dessous sont produites par une exécution réelle du
pipeline sur le sous-ensemble **FD001** de C-MAPSS (100 moteurs
d'entraînement, 100 moteurs de test) — voir
[`reports/technical_report.md`](reports/technical_report.md) pour le détail
complet et les figures.

| Famille | Meilleur modèle | Métrique | Valeur |
|---|---|---|---|
| Supervisé (régression RUL) | Random Forest Regressor | RMSE | 18.74 cycles |
| Supervisé (classification "maintenance bientôt") | Random Forest Classifier | F1 | 0.870 |
| Non supervisé (détection d'anomalies) | Isolation Forest | AUC | 0.892 |
| Deep learning (LSTM, séries temporelles) | LSTM (1 couche, CPU, 15 epochs) | RMSE | 18.98 cycles |

## Ce que ce projet démontre (et ce qu'il ne démontre pas)

**Démontre** : traitement et nettoyage de données en Python (pandas),
feature engineering sur séries temporelles multivariées, machine learning
supervisé (régression et classification, scikit-learn), apprentissage non
supervisé et détection d'anomalies (Isolation Forest, PCA), deep learning
sur CPU avec PyTorch (LSTM), analyse de séries temporelles, visualisation de
données (matplotlib/seaborn), calcul et interprétation honnête de vraies
métriques (RMSE, precision/recall/F1, AUC, corrélation), rédaction d'un
rapport technique structuré, tests automatisés, développement réalisé sous
Linux/macOS en ligne de commande.

**Ne démontre PAS** (à ne pas sur-vendre sur un CV) : traitement Big
Data/Spark (le jeu de données tient en mémoire, ~12 Mo), déploiement en
production (pas d'API de service, pas de monitoring de dérive), publication
scientifique évaluée par des pairs (ce rapport est un document technique
interne, pas un article révisé), recherche d'hyperparamètres exhaustive
(valeurs raisonnables issues de la littérature, pas de grid search
systématique à grande échelle), traitement de données de capteurs réelles
(C-MAPSS est une simulation NASA).

## Limites connues / pistes d'amélioration

Voir la section 6 de [`reports/technical_report.md`](reports/technical_report.md)
pour le détail complet (dataset simulé, sous-ensemble le plus simple choisi
volontairement, modèle de deep learning volontairement réduit, PCA peu
performante sur ce jeu de données, pas de recherche d'hyperparamètres
exhaustive, pas de déploiement).
