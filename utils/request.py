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
df_train_copy['Valencia_pressure'] = df_train_copy['Valencia_pressure'].fillna(df_train_copy['Valencia_pressure'].mode()[0])
df_test_copy['Valencia_pressure'] = df_test_copy['Valencia_pressure'].fillna(df_test_copy['Valencia_pressure'].mode()[0])
df_train_copy['time'] = pd.to_datetime(df_train_copy['time'])
df_test_copy['time'] = pd.to_datetime(df_test_copy['time'])
df_train_copy['Year'] = df_train_copy['time'].dt.year
df_train_copy['Month'] = df_train_copy['time'].dt.month
df_train_copy['Weekday'] = df_train_copy['time'].dt.weekday
df_train_copy['Day'] = df_train_copy['time'].dt.day
df_train_copy['Hour'] = df_train_copy['time'].dt.hour
 
df_test_copy['Year'] = df_test_copy['time'].dt.year
df_test_copy['Month'] = df_test_copy['time'].dt.month
df_test_copy['Weekday'] = df_test_copy['time'].dt.weekday
df_test_copy['Day'] = df_test_copy['time'].dt.day
df_test_copy['Hour'] = df_test_copy['time'].dt.hour
  
df_train_copy.drop('time', axis = 1, inplace = True)
df_test_copy.drop('time', axis = 1, inplace = True)
 
df_train_copy['Valencia_wind_deg'] = df_train_copy['Valencia_wind_deg'].str.extract('(\d+)')
df_train_copy['Valencia_wind_deg'] = pd.to_numeric(df_train_copy['Valencia_wind_deg'])
 
df_test_copy['Valencia_wind_deg'] = df_test_copy['Valencia_wind_deg'].str.extract('(\d+)')
df_test_copy['Valencia_wind_deg'] = pd.to_numeric(df_test_copy['Valencia_wind_deg'])
 
df_train_copy['Seville_pressure'] = df_train_copy['Seville_pressure'].str.extract('(\d+)')
df_train_copy['Seville_pressure'] = pd.to_numeric(df_train_copy['Seville_pressure'])
 
df_test_copy['Seville_pressure'] = df_test_copy['Seville_pressure'].str.extract('(\d+)')
df_test_copy['Seville_pressure'] = pd.to_numeric(df_test_copy['Seville_pressure'])
 
df_train_copy = df_train_copy.drop(['Unnamed: 0'], axis = 1) # axis =1 tells it to drop the column, vertically
df_test_copy = df_test_copy.drop(['Unnamed: 0'], axis = 1) # axis =1 tells it to drop the column, vertically
 
df_train_copy = df_train_copy[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h', 'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id', 'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min','Year', 'Month', 'Day', 'Hour', 'Weekday', 'load_shortfall_3h']]
df_test_copy = df_test_copy[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h', 'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id', 'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min','Year', 'Month', 'Day', 'Hour', 'Weekday']]    

# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = df_test_copy.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://34.245.115.238:5000/api_v0.1'

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
