"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Barcelona_temp', 'Barcelona_temp_max', 'Barcelona_temp_min',
       'Barcelona_wind_deg', 'Barcelona_wind_speed', 'Bilbao_clouds_all',
       'Bilbao_pressure', 'Bilbao_temp', 'Bilbao_temp_max', 'Bilbao_temp_min',
       'Bilbao_wind_deg', 'Madrid_clouds_all', 'Madrid_humidity',
       'Madrid_pressure', 'Madrid_temp', 'Madrid_temp_max', 'Madrid_temp_min',
       'Madrid_wind_speed', 'Seville_clouds_all', 'Seville_humidity',
       'Seville_temp', 'Seville_temp_max', 'Seville_temp_min',
       'Seville_wind_speed', 'Valencia_humidity', 'Valencia_temp',
       'Valencia_temp_max', 'Valencia_temp_min']]

# Fit model
lm_regression = LinearRegression()
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/itumelengk_mlr_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
