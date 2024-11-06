#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Encode categorical columns
def encode_data(data):
    """
    Encode categorical columns in the dataset using Label Encoding.
    """
    # Initialize LabelEncoders
    label_encoders = {
        'origin_airport': LabelEncoder(),
        'destination_airport': LabelEncoder(),
        'departure_date': LabelEncoder(),
        'departure_time_HHMM': LabelEncoder(),
        'cabin_type': LabelEncoder()
    }
    
    # Apply encoding and save encoders for each column
    for column, encoder in label_encoders.items():
        data[column] = encoder.fit_transform(data[column])
    
    return data, label_encoders

