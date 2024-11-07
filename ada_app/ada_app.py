import streamlit as st
import requests
import pandas as pd
import joblib
from datetime import datetime, date
import time

# Load the label encoders
origin_encoder = joblib.load('processed/origin_encoder.joblib')
destination_encoder = joblib.load('processed/destination_encoder.joblib')
cabin_encoder = joblib.load('processed/cabin_encoder.joblib')
ada_pipe = joblib.load('models/ada_reg1.joblib')

# Define the minimum and maximum dates
min_date = date(2011, 2, 6)
max_date = date(2016, 5, 8)

date_range = pd.date_range(start=min_date, end=max_date).to_list()

st.title('ML Streamlit App for Flight Ticket Prices')

st.write("""
## The Machine Learning App

You can use this application to get predictions from a trained Adaboost model for flight tickets for cities in America.
""")

# Function to generate 30-minute intervals
def generate_time_intervals(start='00:00', end='23:30', interval_minutes=30):
    times = pd.date_range(start=start, end=end, freq=f'{interval_minutes}T').time
    return [time.strftime('%H:%M') for time in times]

# Generate time intervals
time_intervals = generate_time_intervals()

# Streamlit selectboxes
flight_date = st.selectbox("Select a date", options=date_range, format_func=lambda x: x.strftime('%Y-%m-%d'))
origin_airport = st.selectbox('origin_airport', origin_encoder.classes_)
destination_airport = st.selectbox('destination_airport', destination_encoder.classes_)
cabin_type = st.selectbox('cabin_type', cabin_encoder.classes_)
departure_time = st.selectbox('Select departure time', options=time_intervals)

# Function to combine date and time into epoch time
def combine_date_time_to_epoch(flight_date, flight_time):
    combined_datetime = pd.to_datetime(f"{flight_date} {flight_time}")
    epoch_time = int(time.mktime(combined_datetime.timetuple()))
    return epoch_time

# Function to convert time string to epoch seconds
def convert_to_epoch_seconds(time_str):
    try:
        # Convert to datetime
        time = pd.to_datetime(time_str, utc=True, errors='coerce')
        # Round to the nearest 30 minutes
        time = (time.floor('30T') + pd.Timedelta(minutes=15)).floor('30T')
        # Convert to epoch seconds
        return int(time.timestamp())
    except Exception as e:
        # Handle any errors and return None or a default value
        return None

# Encode categorical columns and apply convert_to_epoch_seconds function
def encode_data(data):
    """
    Encode categorical columns in the dataset using Label Encoding.
    Apply convert_to_epoch_seconds function to 'departure_time' column.
    """
    # Apply encoding using loaded encoders
    data['origin_airport'] = origin_encoder.transform(data['origin_airport'])
    data['destination_airport'] = destination_encoder.transform(data['destination_airport'])
    data['cabin_type'] = cabin_encoder.transform(data['cabin_type'])
    
    # Apply convert_to_epoch_seconds function to 'departure_time' column
    data['departure_time_seconds'] = data['departure_time'].apply(convert_to_epoch_seconds)
    
    return data

# Combine date and time into a single string
departure_datetime_str = f"{flight_date.strftime('%Y-%m-%d')} {departure_time}"

# Create a DataFrame with the selected values
data = pd.DataFrame({
    'origin_airport': [origin_airport],
    'destination_airport': [destination_airport],
    'cabin_type': [cabin_type],
    'departure_time': [departure_datetime_str]
})

# Encode the data
encoded_data = encode_data(data)

# Define the FastAPI endpoint for flight price predictions
api_url_flight_price = "https://your-api-endpoint.com/predict/flight_price"

# Prepare the input data for the API request
input_data = {
    "origin_airport": int(encoded_data['origin_airport'][0]),
    "destination_airport": int(encoded_data['destination_airport'][0]),
    "cabin_type": int(encoded_data['cabin_type'][0]),
    "departure_time_seconds": int(encoded_data['departure_time_seconds'][0])
}

if st.button('Predict Flight Price'):
    # Make the API request for flight price prediction
    response = requests.get(api_url_flight_price, params=input_data)
    st.write(f"Status Code: {response.status_code}")
    st.write(f"Response Headers: {response.headers}")
    st.write(f"Response Text: {response.text}")
    
    if response.status_code == 200:
        prediction = response.json()
        st.write("Predicted Flight Price:")
        st.write(prediction)
    else:
        st.write("Error: Unable to fetch prediction")
