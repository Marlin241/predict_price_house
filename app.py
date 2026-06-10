import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd


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

    longitude = st.sidebar.slider("Longitude", -124.3, -114.3, -119.6)
    latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 35.6)
    housing_median_age = st.sidebar.slider("Âge médian des maisons", 1, 52, 29)
    population = st.sidebar.slider("Population du quartier", 3, 3500, 1425)
    median_income = st.sidebar.slider("Revenu médian (dizaines de milliers $)", 0.5, 15.0, 3.87)
    avg_rooms = st.sidebar.slider("Nb moyen de pièces par ménage", 1.0, 10.0, 5.4)
    avg_bedrooms = st.sidebar.slider("Nb moyen de chambres par ménage", 1.0, 3.0, 1.1)
    avg_occupancy = st.sidebar.slider("Occupants moyens par ménage", 1.0, 6.0, 3.1)
    ocean_proximity = st.sidebar.selectbox(
        "Proximité de l'océan",
        ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN']
    )

    features = compute_features_for_prediction(
        longitude, latitude, housing_median_age, population, median_income,
        avg_rooms, avg_bedrooms, avg_occupancy, ocean_proximity
    )
    X = pd.DataFrame([features], columns=feature_names)

    if st.sidebar.button("🔮 Estimer le prix"):
        prediction = pipeline.predict(X)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("💰 Prix estimé", f"${prediction:,.0f}")

        with col2:
            st.subheader("Performances du modèle")
            st.write(f"**Modèle :** {metrics['model_name']}")
            st.write(f"**R²** : {metrics['r2']:.3f}")
            st.write(f"**RMSE** : ${metrics['rmse']:,.0f}")
            st.write(f"**MAE** : ${metrics['mae']:,.0f}")

        st.subheader("Top 3 features les plus influentes")
        coefs = np.abs(pipeline.named_steps['model'].coef_)
        top_idx = np.argsort(coefs)[::-1][:3]
        for i in top_idx:
            st.write(f"• **{feature_names[i]}** : {coefs[i]:.4f}")


if __name__ == "__main__":
    main()
