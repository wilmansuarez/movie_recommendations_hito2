def compute_prediction_accuracy(model, test_df):
    # e.g., precision@k or RMSE
    pass

def compute_training_time(model_class, df):
    # e.g., time how long fit() takes
    pass

def compute_inference_time(model, user_id):
    # e.g., time how long recommend(user_id) takes
    pass

def compute_model_size(model_path):
    import os
    return os.path.getsize(model_path)
