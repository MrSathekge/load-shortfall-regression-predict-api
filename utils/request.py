"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Kaggle challenge.
#test = pd.read_csv('./data/df_test.csv')
df_train = pd.read_csv('./data/df_train.csv')
df_test = pd.read_csv('./data/df_test.csv')
    
df_train_copy = df_train.copy()
df_test_copy = df_test.copy()
    
df_train_copy = df_train_copy.drop(['Unnamed: 0', 'Valencia_pressure', 'time','Seville_pressure', 'Valencia_wind_deg'], axis = 1) # axis =1 tells it to drop the column, vertically
df_test_copy = df_test_copy.drop(['Unnamed: 0', 'Valencia_pressure', 'time','Seville_pressure', 'Valencia_wind_deg'], axis = 1)
       
df_train_copy = df_train_copy[['Madrid_wind_speed', 'Bilbao_rain_1h', 'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id', 'Seville_temp_max', 'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min', 'load_shortfall_3h']]
df_test_copy = df_test_copy[['Madrid_wind_speed', 'Bilbao_rain_1h', 'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id', 'Seville_temp_max', 'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min']]    

# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = df_test_copy.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://127.0.0.1:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {df_test_copy.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50) 
