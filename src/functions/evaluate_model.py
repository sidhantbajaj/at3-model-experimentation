#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    Predict and evaluate the model on test data.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Mean Absolute Error:", mae)
    print("R² Score:", r2)
    print("RMSE:", rmse)
    
    return mae, r2, rmse

