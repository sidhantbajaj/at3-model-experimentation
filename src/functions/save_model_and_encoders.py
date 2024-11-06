#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Save model and encoders
def save_model_and_encoders(model, label_encoders, model_path, encoders_path_prefix):
    """
    Save the trained model and label encoders.
    """
    # Save model with compression
    joblib.dump(model, model_path, compress=3)
    print(f"Model saved at {model_path}")
    
    # Save each label encoder
    for column, encoder in label_encoders.items():
        encoder_path = f"{encoders_path_prefix}_{column}_label_encoder.pkl"
        joblib.dump(encoder, encoder_path)
        print(f"{column} encoder saved at {encoder_path}")

