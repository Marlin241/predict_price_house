# app.py


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
