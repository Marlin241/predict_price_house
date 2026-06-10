# Prédiction du Prix de l'Immobilier — Californie

Projet de Machine Learning supervisé : prédiction du prix médian des logements en Californie à partir du dataset [Kaggle California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data).

## Aperçu

| | |
|---|---|
| **Type** | Régression supervisée |
| **Meilleur modèle** | LinearRegression |
| **R²** | 0.591 |
| **RMSE** | $73,455 |
| **MAE** | $52,377 |
| **Données** | 20 640 logements · 12 features |

## Fonctionnalités

- **Notebook d'analyse** : exploration, visualisations (heatmap de corrélation, histogrammes, résidus), feature engineering, comparaison de 3 modèles de régression
- **Application Streamlit** : estimation en temps réel via un formulaire interactif (ville, ancienneté, revenus, nombre de pièces…)
- **Tests unitaires** : 6 tests pytest pour la fonction de préparation des features

## Structure du projet

```
├── notebook.ipynb          # Analyse complète (6 sections)
├── app.py                  # Application Streamlit
├── requirements.txt        # Dépendances Python
├── model.pkl               # Pipeline entraîné (LinearRegression + StandardScaler)
├── feature_names.pkl       # Noms des 12 features
├── metrics.json            # Métriques du meilleur modèle
├── tests/
│   └── test_features.py    # Tests unitaires (pytest)
└── archive/
    └── housing.csv         # Dataset Kaggle
```

## Dataset

**Source :** [California Housing Prices — Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)

| Colonne | Description |
|---|---|
| `longitude` / `latitude` | Coordonnées géographiques du quartier |
| `housing_median_age` | Âge médian des logements |
| `total_rooms` / `total_bedrooms` | Totaux par bloc (207 NaN imputés par la médiane) |
| `population` | Population du bloc |
| `households` | Nombre de ménages |
| `median_income` | Revenu médian (en dizaines de milliers $) |
| `ocean_proximity` | Catégorie de proximité à l'océan |
| `median_house_value` | **Cible** — prix médian en dollars |

## Feature Engineering

4 features dérivées calculées à partir des totaux et du nombre de ménages :

| Feature | Formule |
|---|---|
| `avg_rooms` | `total_rooms / households` |
| `avg_bedrooms` | `total_bedrooms / households` |
| `avg_occupancy` | `population / households` |
| `rooms_per_bedroom` | `avg_rooms / avg_bedrooms` |

`ocean_proximity` est encodée en one-hot (référence : `<1H OCEAN`), ce qui donne **12 features finales**.

## Modèles comparés

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| **LinearRegression** | **$73,455** | **$52,377** | **0.5913** |
| RidgeCV | $73,455 | $52,376 | 0.5913 |
| LassoCV | $73,494 | $52,338 | 0.5909 |

Chaque modèle est encapsulé dans un `Pipeline(StandardScaler → modèle)` scikit-learn.

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Marlin241/predict_price_house.git
cd predict_price_house

# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Lancer l'application Streamlit

```bash
streamlit run app.py
```

Sélectionnez une ville californienne, ajustez les caractéristiques du bien dans la barre latérale, puis cliquez sur **Estimer le prix** pour obtenir :
- Une fourchette de prix (estimation ± erreur moyenne)
- La fiabilité du modèle (R² en %)
- Un graphique des facteurs les plus influents

### Consulter le notebook

```bash
jupyter notebook notebook.ipynb
```

Le notebook est organisé en 6 sections :

1. **Chargement & aperçu** — `pd.read_csv`, statistiques descriptives, valeurs manquantes
2. **Analyse exploratoire (EDA)** — histogrammes, heatmap de corrélation, scatter plots
3. **Nettoyage & Feature Engineering** — imputation, features dérivées, one-hot encoding
4. **Modélisation** — entraînement et comparaison LinearRegression / Ridge / Lasso
5. **Analyse des résidus** — scatter résidus vs prédictions, détection des outliers
6. **Export** — sauvegarde du pipeline, des feature names et des métriques

### Lancer les tests

```bash
pytest tests/ -v
```

## Technologies

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![pandas](https://img.shields.io/badge/pandas-2.0+-green)
