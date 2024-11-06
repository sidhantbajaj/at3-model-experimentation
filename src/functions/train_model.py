#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Train model
def train_model(X_train, y_train):
    """
    Train a Random Forest model with specified parameters.
    """
    # Initialize Random Forest with parameters for reduced model size
    model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_leaf=2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

