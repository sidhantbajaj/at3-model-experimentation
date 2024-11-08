#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib


# In[ ]:


# Load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load the dataset, rename relevant columns, and select only the necessary columns.
    """
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Rename columns for clarity
    data = data.rename(columns={
        'startingAirport': 'origin_airport', 
        'destinationAirport': 'destination_airport',
        'flightDate': 'departure_date', 
        'totalFare': 'fare'
    })
    
    # Extract departure time in HH:MM format
    data['departure_time_HHMM'] = pd.to_datetime(
        data['segmentsDepartureTimeRaw'].apply(lambda x: x.split('||')[0]),
        utc=True,
        errors='coerce'
    ).dt.strftime('%H:%M')
    
    # Extract cabin type
    data['cabin_type'] = data['segmentsCabinCode'].apply(lambda x: x.split('||')[0])
    
    # Select and return only the relevant columns
    data = data[['origin_airport', 'destination_airport', 'departure_date', 'departure_time_HHMM', 'cabin_type', 'fare']]
    return data

