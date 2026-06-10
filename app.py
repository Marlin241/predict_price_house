import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VILLES_CALIFORNIE = {
    'Los Angeles':   (-118.24, 34.05),
    'San Francisco': (-122.42, 37.77),
    'San Diego':     (-117.16, 32.72),
    'Sacramento':    (-121.49, 38.58),
    'San Jose':      (-121.89, 37.34),
    'Fresno':        (-119.79, 36.74),
    'Oakland':       (-122.27, 37.80),
    'Bakersfield':   (-119.02, 35.37),
    'Long Beach':    (-118.19, 33.77),
    'Santa Barbara': (-119.70, 34.42),
}

LABELS_FR = {
    'median_income':            'Revenu médian du quartier',
    'longitude':                'Localisation (longitude)',
    'latitude':                 'Localisation (latitude)',
    'housing_median_age':       'Ancienneté des logements',
    'avg_rooms':                'Nb moyen de pièces',
    'avg_bedrooms':             'Nb moyen de chambres',
    'avg_occupancy':            'Occupation moyenne',
    'rooms_per_bedroom':        'Ratio pièces / chambres',
    'population':               'Population du quartier',
    'ocean_proximity_INLAND':   'Zone : intérieur des terres',
    'ocean_proximity_NEAR BAY': "Zone : près d'une baie",
    'ocean_proximity_NEAR OCEAN': 'Zone : bord de mer',
}


def compute_features_for_prediction(longitude, latitude, housing_median_age, population,
                                     median_income, avg_rooms, avg_bedrooms, avg_occupancy,
                                     ocean_proximity):
    rooms_per_bedroom = avg_rooms / avg_bedrooms
    # One-hot encoding — référence : <1H OCEAN (toutes les dummies = 0)
    ocean_inland = 1 if ocean_proximity == 'INLAND' else 0
    ocean_near_bay = 1 if ocean_proximity == 'NEAR BAY' else 0
    ocean_near_ocean = 1 if ocean_proximity == 'NEAR OCEAN' else 0
    return [
        longitude, latitude, housing_median_age, population, median_income,
        avg_rooms, avg_bedrooms, avg_occupancy, rooms_per_bedroom,
        ocean_inland, ocean_near_bay, ocean_near_ocean
    ]


@st.cache_resource
def load_artifacts():
    # Chargement local uniquement — ne jamais charger un .pkl depuis une URL externe.
    pipeline = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    metrics = json.load(open("metrics.json"))
    return pipeline, feature_names, metrics


def main():
    st.title("🏠 Estimation du prix immobilier en Californie")
    pipeline, feature_names, metrics = load_artifacts()

    st.sidebar.header("Caractéristiques du bien")

    ville = st.sidebar.selectbox("📍 Ville", list(VILLES_CALIFORNIE.keys()))
    longitude, latitude = VILLES_CALIFORNIE[ville]

    housing_median_age = st.sidebar.slider("🏗️ Ancienneté du quartier (années)", 1, 52, 29)
    population = st.sidebar.slider("👥 Population du quartier", 3, 3500, 1425)
    median_income = st.sidebar.slider(
        "💼 Revenu médian du quartier (k$)",
        0.5, 15.0, 3.87,
        help="Revenu médian des ménages du quartier, en dizaines de milliers de dollars. Ex : 5.0 = 50 000 $/an."
    )
    avg_rooms = st.sidebar.slider("🚪 Nb moyen de pièces par logement", 1.0, 10.0, 5.4)
    avg_bedrooms = st.sidebar.slider("🛏️ Nb moyen de chambres par logement", 1.0, 3.0, 1.1)
    avg_occupancy = st.sidebar.slider("👤 Occupants moyens par logement", 1.0, 6.0, 3.1)
    ocean_proximity = st.sidebar.selectbox(
        "🌊 Proximité de l'océan",
        ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN'],
        help=(
            "**<1H OCEAN** : à moins d'1h de l'océan\n\n"
            "**INLAND** : intérieur des terres\n\n"
            "**NEAR BAY** : près d'une baie (ex. San Francisco Bay)\n\n"
            "**NEAR OCEAN** : directement en bord de mer"
        )
    )

    features = compute_features_for_prediction(
        longitude, latitude, housing_median_age, population, median_income,
        avg_rooms, avg_bedrooms, avg_occupancy, ocean_proximity
    )
    X = pd.DataFrame([features], columns=feature_names)

    if st.sidebar.button("🔮 Estimer le prix"):
        prediction = pipeline.predict(X)[0]
        mae = metrics['mae']

        # --- Prix estimé avec fourchette ---
        st.subheader("💰 Prix estimé")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fourchette basse", f"${max(0, prediction - mae):,.0f}")
        with col2:
            st.metric("Estimation centrale", f"${prediction:,.0f}")
        with col3:
            st.metric("Fourchette haute", f"${prediction + mae:,.0f}")
        st.caption(
            f"La fourchette correspond à l'erreur absolue moyenne du modèle (±${mae:,.0f}). "
            "Le prix réel se situe généralement dans cet intervalle."
        )

        st.divider()

        # --- Fiabilité du modèle ---
        st.subheader("📊 Fiabilité du modèle")
        r2_pct = int(metrics['r2'] * 100)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Précision globale",
                f"{r2_pct} %",
                help="Coefficient de détermination R² : part de la variation des prix expliquée par le modèle."
            )
            st.progress(metrics['r2'])
            st.caption(f"Le modèle explique **{r2_pct}%** de la variation des prix sur les données de test.")
        with col2:
            st.metric("Modèle utilisé", metrics['model_name'])
            st.metric(
                "Erreur typique",
                f"±${metrics['rmse']:,.0f}",
                help="RMSE : écart-type des erreurs de prédiction. Plus cette valeur est faible, plus le modèle est précis."
            )
            st.caption("En moyenne, l'estimation s'écarte du prix réel de ±${:,.0f}.".format(metrics['mae']))

        st.divider()

        # --- Facteurs influents ---
        st.subheader("🔍 Facteurs les plus influents sur le prix")
        coefs = np.abs(pipeline.named_steps['model'].coef_)
        top_idx = np.argsort(coefs)[::-1][:5]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = coefs[top_idx]
        top_pct = top_vals / top_vals.sum() * 100

        labels = [LABELS_FR.get(n, n) for n in top_names]
        fig, ax = plt.subplots(figsize=(7, 2.8))
        bars = ax.barh(labels[::-1], top_pct[::-1], color='steelblue', height=0.6)
        ax.bar_label(bars, fmt='%.1f%%', padding=5, fontsize=10)
        ax.set_xlabel("Influence relative (%)")
        ax.set_xlim(0, top_pct[0] * 1.25)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Ce graphique montre le poids relatif de chaque facteur dans la décision du modèle. "
            "Un facteur à 60% pèse deux fois plus qu'un facteur à 30%."
        )


if __name__ == "__main__":
    main()
