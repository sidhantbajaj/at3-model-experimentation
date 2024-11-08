#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Split the data
def split_data(data, test_size=0.2, sample_fraction=0.1, random_state=42):
    """
    Sample a fraction of the data, separate features and target, and split into train/test sets.
    """
    # Take a random sample for quicker training
    sampled_data = data.sample(frac=sample_fraction, random_state=random_state)
    
    # Define features and target
    X = sampled_data[['origin_airport', 'destination_airport', 'departure_date', 'departure_time_HHMM', 'cabin_type']]
    y = sampled_data['fare']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

