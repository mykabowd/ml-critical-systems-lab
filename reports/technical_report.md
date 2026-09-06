# Rapport technique : maintenance predictive de systemes critiques via le jeu de donnees NASA C-MAPSS

**Auteur** : Maxime Kabowd
**Date** : Septembre 2026
**Code source** : [`ml-critical-systems-lab`](..) (depot local, voir `README.md`)
**Reproductibilite** : toutes les metriques de ce rapport sont produites par
`python -m ml_critical_systems_lab.pipeline` (aucune valeur inventee ou
extrapolee) et exportees vers `reports/results.json`.

---

## 1. Problematique

Les systemes critiques (moteurs d'aeronefs, systemes embarques de defense,
capteurs de plateformes complexes) doivent etre maintenus selon des
contraintes de securite et de disponibilite tres strictes. Deux strategies
de maintenance s'opposent traditionnellement :

- **maintenance corrective** : on repare apres la panne — inacceptable pour
  un systeme critique (risque pour la securite, indisponibilite non
  planifiee) ;
- **maintenance preventive systematique** : on remplace des pieces a
  intervalles fixes, independamment de leur etat reel — couteux et parfois
  inutile.

La **maintenance predictive** cherche un compromis : estimer, a partir de
mesures de capteurs, la duree de vie residuelle (**RUL — Remaining Useful
Life**) d'un composant, afin de planifier une intervention avant la panne
mais sans gaspiller de duree de vie utile. C'est un probleme central pour
des acteurs comme Thales, ou la fiabilite et la disponibilite des systemes
(aeronautique, defense, systemes de transport) sont des exigences de
premier ordre.

Ce projet applique trois familles de methodes de machine learning a ce
probleme sur un jeu de donnees public de simulation de degradation de
moteurs d'avion (turbofan), afin de :

1. estimer le RUL par regression (ML supervise) ;
2. detecter automatiquement un etat degrade sans etiquette de panne
   (apprentissage non supervise / detection d'anomalies) ;
3. exploiter la nature sequentielle des mesures de capteurs via un modele de
   deep learning (LSTM) operant sur des fenetres temporelles.

## 2. Description du dataset

**NASA C-MAPSS — Turbofan Engine Degradation Simulation Data Set**
(Commercial Modular Aero-Propulsion System Simulation), publie par la NASA
Prognostics Center of Excellence.

> Reference : A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage
> Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*,
> Proceedings of the 1st International Conference on Prognostics and Health
> Management (PHM08), Denver, CO, octobre 2008.
>
> Source de telechargement utilisee :
> `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`

Le jeu de donnees complet comporte 4 sous-ensembles (FD001 a FD004), qui
different par le nombre de conditions operationnelles et de modes de panne
simules. Ce projet utilise exclusivement le sous-ensemble **FD001**
(**une seule condition operationnelle, un seul mode de panne : degradation
HPC — High Pressure Compressor**), le plus simple et le plus propre du jeu
de donnees, choisi deliberement pour une demonstration methodologique claire
plutot que pour maximiser la difficulte du probleme.

Chaque ligne represente un cycle operationnel d'un moteur, avec 26 colonnes :
numero d'unite, numero de cycle, 3 reglages operationnels, et 21 mesures de
capteurs (temperatures, pressions, vitesses de rotation, debits, etc.).

| Fichier | Contenu | Dimensions reelles |
|---|---|---|
| `train_FD001.txt` | 100 moteurs, trajectoires completes jusqu'a la panne | 20 631 lignes |
| `test_FD001.txt` | 100 moteurs, trajectoires tronquees avant la panne | 13 096 lignes |
| `RUL_FD001.txt` | RUL reel (verite terrain) au dernier cycle observe de chaque moteur de test | 100 valeurs |

Chaque moteur commence dans un etat normal (avec une usure initiale variable
et non renseignee, consideree comme non-anormale) et developpe une panne qui
s'aggrave jusqu'a la fin de la trajectoire (train) ou jusqu'a un point
arbitraire avant la panne (test).

## 3. Methodologie

### 3.1 Pretraitement et feature engineering (`data.py`, `features.py`)

1. **Chargement** : fichiers texte separes par des espaces, noms de colonnes
   explicites assignes manuellement (`unit_number`, `time_in_cycles`,
   `op_setting_1..3`, `sensor_1..21`).
2. **Calcul du label RUL** (jeu d'entrainement uniquement) :
   `RUL(unite, cycle) = max_cycle(unite) - cycle`, **plafonne a 125 cycles**.
   Ce plafond est une convention standard de la litterature C-MAPSS (Heimes,
   2008 ; Saxena & Goebel, 2008) : la degradation n'est en pratique
   significative que dans les derniers cycles avant la panne, et sans
   plafond le modele serait penalise pour ne pas predire une degradation
   lineaire des les premiers cycles, alors qu'aucun signal de degradation
   n'est encore present dans les mesures a ce stade.
3. **Label de classification binaire** : `maintenance_soon = 1` si
   `RUL < 30` cycles, `0` sinon (seuil arbitraire mais realiste — declencher
   une planification de maintenance environ un mois-machine avant la panne
   estimee).
4. **Statistiques glissantes** (feature engineering) : moyenne et
   ecart-type glissants (fenetre de 5 cycles) sur 9 capteurs a variance
   non-nulle, identifies par inspection statistique du jeu d'entrainement
   (`sensor_2, 3, 4, 7, 11, 12, 15, 20, 21` — les 12 autres capteurs et
   `op_setting_3` sont quasi constants sur FD001 dans ce sous-ensemble a
   condition operationnelle unique, et n'apportent donc pas d'information).
   Calcul effectue **par unite** (`groupby("unit_number")`) pour ne jamais
   melanger les cycles de deux moteurs differents.
5. **Normalisation** : `StandardScaler` de scikit-learn, **ajuste
   uniquement sur le jeu d'entrainement** puis applique au jeu de test
   (`fit` sur train, `transform` sur train et test), afin d'eviter toute
   fuite d'information (data leakage) du test vers l'entrainement.

Au total, **42 colonnes de features** : 3 reglages operationnels + 21
capteurs bruts + 18 statistiques glissantes (moyenne/ecart-type sur 9
capteurs).

### 3.2 Separation train / test

Le jeu de donnees C-MAPSS fournit une separation train/test **par moteur**
(pas par ligne) : 100 moteurs d'entrainement (trajectoires completes) et 100
moteurs de test (trajectoires tronquees, avec RUL reel fourni uniquement au
dernier cycle observe). Cette separation, deja fournie par la NASA, est
utilisee telle quelle : aucune fuite d'information n'est possible puisque
les moteurs de test n'apparaissent jamais dans le jeu d'entrainement.

Pour les modeles supervises et le LSTM, l'evaluation finale se fait sur le
**dernier cycle observe de chaque moteur de test**, comparee au RUL reel
fourni par `RUL_FD001.txt` — c'est la definition officielle du benchmark
C-MAPSS.

Pour la detection d'anomalies non supervisee (qui n'a pas de jeu de test
officiel puisqu'elle n'utilise pas d'etiquette), le jeu d'entrainement est
lui-meme separe **par moteur** en un sous-ensemble d'ajustement (80 moteurs)
et un sous-ensemble d'evaluation (20 moteurs, jamais vus par le detecteur),
afin d'evaluer honnetement la generalisation du score d'anomalie a des
moteurs non vus, tout en utilisant le RUL connu de ces moteurs (jamais donne
au modele) pour l'evaluation a posteriori.

## 4. Les trois familles de modeles et leurs resultats reels

Tous les chiffres ci-dessous proviennent de l'execution reelle du pipeline
(`reports/results.json`), sans aucune valeur inventee.

### 4.1 Machine learning supervise (`supervised.py`)

**(a) Regression du RUL** — comparaison Regression lineaire vs Random
Forest Regressor (200 arbres, profondeur max. 10), evaluees sur le dernier
cycle observe des 100 moteurs de test :

| Modele | RMSE (cycles) | MAE (cycles) | R² |
|---|---|---|---|
| Regression lineaire | 22.26 | 17.87 | 0.713 |
| **Random Forest Regressor** | **18.74** | **13.40** | **0.797** |

Le Random Forest reduit le RMSE de ~16% par rapport a la regression
lineaire, ce qui est attendu : la relation entre mesures capteurs et RUL
n'est pas lineaire (la degradation s'accelere en fin de vie).

**(b) Classification binaire "maintenance necessaire bientot"**
(`RUL < 30` cycles) — Logistic Regression vs Random Forest Classifier
(`class_weight="balanced"` pour compenser le desequilibre des classes) :

| Modele | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.808 | 0.840 | 0.824 | 0.910 |
| **Random Forest Classifier** | **0.952** | 0.800 | **0.870** | **0.940** |

Matrice de confusion (Random Forest Classifier, 100 moteurs de test) :

| | Predit : normal | Predit : maintenance bientot |
|---|---|---|
| **Reel : normal** | 74 | 1 |
| **Reel : maintenance bientot** | 5 | 20 |

Le Random Forest Classifier n'a qu'**1 faux positif** sur 75 moteurs
normaux et **5 faux negatifs** sur 25 moteurs proches de la panne — un
compromis raisonnable pour un cas d'usage de planification de maintenance,
ou un faux negatif (panne non anticipee) est plus couteux qu'un faux
positif (verification superflue).

Figure : `figures/regression_rf_scatter.png`, `figures/regression_linear_scatter.png`,
`figures/classification_confusion_matrix.png`.

### 4.2 Apprentissage non supervise / detection d'anomalies (`unsupervised.py`)

Isolation Forest et erreur de reconstruction PCA (5 composantes),
**ajustes sans aucune etiquette** sur les mesures capteurs normalisees de 80
moteurs d'entrainement, puis **evalues sur 20 moteurs jamais vus** en
utilisant leur RUL connu (uniquement pour la mesure de performance, pas pour
l'entrainement) :

| Methode | AUC (score anomalie vs "proche panne") | Correlation de Spearman (score, RUL) |
|---|---|---|
| **Isolation Forest** | **0.892** | **-0.277** |
| PCA (erreur de reconstruction, 5 composantes, 91.1% variance expliquee) | 0.511 | -0.016 |

**Isolation Forest** discrimine bien les etats "proches de la panne"
(RUL < 30) des etats normaux (AUC = 0.892, nettement superieur au hasard =
0.5), et son score augmente globalement lorsque le RUL diminue (correlation
negative). **La PCA a 5 composantes, en revanche, ne fait pratiquement pas
mieux que le hasard (AUC = 0.511)** sur ce jeu de donnees : la degradation
des capteurs FD001 se manifeste probablement par des changements de
distribution que 5 composantes principales lineaires ne capturent pas
suffisamment (ou par des correlations entre capteurs davantage que par une
augmentation de variance globale). C'est un resultat honnete et attendu :
les deux methodes ne sont pas interchangeables et Isolation Forest
(non-lineaire, base sur l'isolement d'observations dans l'espace des
features) capture mieux ce signal de degradation ici.

Figure : `figures/anomaly_score_distribution.png` (Isolation Forest).

### 4.3 Deep learning + series temporelles (`deep_learning.py`)

Un LSTM leger (1 couche recurrente, `hidden_size=32`, tete de regression a
2 couches lineaires — **9 313 parametres** au total) est entraine sur des
**fenetres glissantes de 30 cycles** (17 731 sequences d'entrainement
generees a partir des 100 moteurs de train), pour predire le RUL au dernier
cycle de chaque fenetre.

- **Entrainement** : 15 epochs, `Adam` (`lr=1e-3`), MSE loss, batch size 64,
  **CPU uniquement**, **35.5 secondes** au total sur la machine de
  developpement.
- **Perte finale** : train MSE = 60.65, validation MSE = 70.02
  (validation = 20% des sequences d'entrainement, split aleatoire par
  fenetre).
- **Evaluation sur le jeu de test officiel** (dernier cycle observe de
  chaque moteur de test, RUL reel de `RUL_FD001.txt`) :

| Modele | RMSE (cycles) | MAE (cycles) | R² |
|---|---|---|---|
| **LSTM (deep learning)** | **18.98** | 14.11 | 0.791 |

Le LSTM obtient une performance **quasiment identique au Random Forest**
(RMSE 18.98 vs 18.74) malgre une architecture volontairement minimaliste et
seulement 15 epochs d'entrainement — un resultat coherent avec la
litterature C-MAPSS, ou les modeles sequentiels (LSTM, CNN 1D) et les
modeles d'ensemble sur features engineered atteignent des performances
comparables sur FD001 (le sous-ensemble le plus simple du jeu de donnees).

Figure : `figures/lstm_training_curve.png`, `figures/regression_lstm_scatter.png`.

### 4.4 Tableau comparatif de synthese

| Famille | Meilleur modele | Metrique principale | Valeur reelle |
|---|---|---|---|
| Supervise (regression) | Random Forest Regressor | RMSE | **18.74 cycles** |
| Supervise (classification) | Random Forest Classifier | F1 | **0.870** |
| Non supervise (anomalies) | Isolation Forest | AUC | **0.892** |
| Deep learning (series temporelles) | LSTM | RMSE | **18.98 cycles** |

## 5. Figures

Toutes les figures sont generees par `visualize.py` et sauvegardees dans
`reports/figures/` :

1. **Degradation des capteurs** au fil des cycles, 6 moteurs superposes —
   [`figures/sensor_degradation.png`](figures/sensor_degradation.png)
2. **RUL predit vs reel — Random Forest** —
   [`figures/regression_rf_scatter.png`](figures/regression_rf_scatter.png)
3. **RUL predit vs reel — regression lineaire** —
   [`figures/regression_linear_scatter.png`](figures/regression_linear_scatter.png)
4. **RUL predit vs reel — LSTM** —
   [`figures/regression_lstm_scatter.png`](figures/regression_lstm_scatter.png)
5. **Matrice de confusion — classification `maintenance_soon`** —
   [`figures/classification_confusion_matrix.png`](figures/classification_confusion_matrix.png)
6. **Distribution du score d'anomalie** (normal vs proche panne, Isolation
   Forest) —
   [`figures/anomaly_score_distribution.png`](figures/anomaly_score_distribution.png)
7. **Courbe de perte d'entrainement/validation du LSTM** —
   [`figures/lstm_training_curve.png`](figures/lstm_training_curve.png)

## 6. Limites

Ce projet est une demonstration methodologique et non un systeme pret pour
la production. Limites assumees et explicites :

- **Dataset simule** : C-MAPSS est une simulation NASA, pas des donnees de
  capteurs reels d'un systeme en exploitation. Les conclusions ne se
  transposent pas directement a un systeme reel sans validation
  supplementaire.
- **Sous-ensemble le plus simple** (FD001, une seule condition
  operationnelle, un seul mode de panne) : les sous-ensembles FD002/FD004
  (6 conditions operationnelles, jusqu'a 2 modes de panne) sont
  significativement plus difficiles et n'ont pas ete traites ici.
- **Pas de recherche d'hyperparametres exhaustive** : les hyperparametres
  (nombre d'arbres, profondeur, taille du LSTM, fenetre temporelle, seuil de
  classification) ont ete fixes a des valeurs raisonnables issues de la
  litterature et de quelques essais manuels, pas d'une recherche
  systematique (grid/random search, validation croisee k-fold).
- **Modele de deep learning volontairement reduit** : 1 couche LSTM,
  hidden_size=32, 15 epochs, pour s'entrainer en moins d'une minute sur CPU.
  Un modele plus profond et entraine plus longtemps (avec early stopping et
  GPU) obtiendrait probablement de meilleures performances.
- **PCA peu performante ici** (AUC proche du hasard) : cela ne signifie pas
  que la PCA est une mauvaise methode en general, mais qu'elle n'est pas
  adaptee sans modification (ex. PCA non lineaire, autoencodeur, choix
  different du nombre de composantes) a la structure de degradation de ce
  jeu de donnees particulier — un resultat honnete plutot qu'un choix de
  presenter uniquement la methode qui fonctionne bien.
- **Pas de deploiement en production** : aucune API de service, aucun
  monitoring de derive de donnees (data drift), aucune infrastructure
  Big Data/Spark — hors-scope de ce projet, qui est un projet de
  demonstration de recherche appliquee, pas un livrable operationnel.
- **Pas de publication scientifique** : ce document est un rapport
  technique interne, pas un article evalue par des pairs.

## 7. Conclusion et pistes d'amelioration

Ce projet demontre, sur un jeu de donnees public et reconnu de la
litterature en prognostics et sante des systemes (PHM), l'application
correcte des trois grandes familles de machine learning a un probleme de
maintenance predictive : **apprentissage supervise** (regression et
classification, RMSE = 18.74 cycles et F1 = 0.870), **apprentissage non
supervise / detection d'anomalies** (Isolation Forest, AUC = 0.892) et
**deep learning sur series temporelles** (LSTM, RMSE = 18.98 cycles,
comparable au meilleur modele classique malgre une architecture minimale).

Pistes d'amelioration identifiees pendant ce travail :

- Etendre l'analyse aux sous-ensembles FD002/FD003/FD004 (conditions
  operationnelles multiples, plusieurs modes de panne) pour tester la
  robustesse des methodes dans un cadre plus realiste.
- Recherche d'hyperparametres systematique (ex. `RandomizedSearchCV`,
  validation croisee par groupe de moteurs) pour les modeles scikit-learn.
- Explorer un autoencodeur (au lieu de la PCA lineaire) pour la detection
  d'anomalies, potentiellement plus performant sur des relations non
  lineaires entre capteurs.
- Ajouter un mecanisme d'attention ou un CNN 1D en complement/remplacement
  du LSTM, et comparer les performances sur une fenetre temporelle plus
  longue.
- Quantifier l'incertitude des predictions (ex. intervalles de confiance
  bootstrap pour le Random Forest, dropout bayesien pour le LSTM), utile
  pour une decision de maintenance en conditions reelles.
