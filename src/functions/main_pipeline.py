#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Define main pipeline function
def main_pipeline(file_path, model_path, encoders_path_prefix):
    """
    Main pipeline for loading data, preprocessing, training, and saving the model.
    """
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    
    # Encode categorical data
    data, label_encoders = encode_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model and encoders
    save_model_and_encoders(model, label_encoders, model_path, encoders_path_prefix)

