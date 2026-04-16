import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def load_models(model_path="Addiction Level_models.pkl"):
    """Load all trained models from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_model(model_name="RandomForest", model_path="Addiction Level_models.pkl"):
    """Load a specific model by name."""
    models = load_models(model_path)
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    return models[model_name]


def get_training_columns(data_path="data/Time_Wasters_on_Social_Media.csv"):
    """Get the column names used during training."""
    df_sample = pd.read_csv(data_path)
    df_sample = df_sample.drop(["Addiction Level", "ProductivityLoss", "Self Control", "Satisfaction"], axis=1)
    df_sample = pd.get_dummies(df_sample, drop_first=True)
    return df_sample.columns.tolist()


def preprocess_input(input_data, training_columns=None, scale_features=True, data_path="data/Time_Wasters_on_Social_Media.csv"):
    """
    Preprocess input data for prediction.
    
    Args:
        input_data: Dictionary of input features
        training_columns: List of columns used during training
        scale_features: Whether to scale the features
        data_path: Path to the training data CSV
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    if training_columns is None:
        training_columns = get_training_columns(data_path)
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply one-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    # Add missing columns with 0 values
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only training columns in the same order
    df = df[training_columns]
    
    # Scale features if needed
    if scale_features:
        scaler = StandardScaler()
        df_full = pd.read_csv(data_path)
        df_full = df_full.drop(["Addiction Level", "ProductivityLoss", "Self Control", "Satisfaction"], axis=1)
        df_full = pd.get_dummies(df_full, drop_first=True)
        df_full = df_full[training_columns]
        scaler.fit(df_full)
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df


def predict(input_data, model_name="RandomForest", model_path="Addiction Level_models.pkl", 
            scale_features=True, data_path="data/Time_Wasters_on_Social_Media.csv"):
    """
    Make a prediction using a trained model.
    
    Args:
        input_data: Dictionary of input features or preprocessed DataFrame
        model_name: Name of the model to use for prediction
        model_path: Path to the pickled models file
        scale_features: Whether to scale the features
        data_path: Path to the training data CSV
        
    Returns:
        Predicted addiction level (0-5)
    """
    model = load_model(model_name, model_path)
    
    # Preprocess input if it's a dictionary
    if isinstance(input_data, dict):
        X = preprocess_input(input_data, training_columns=None, 
                            scale_features=scale_features, data_path=data_path)
    else:
        X = input_data
    
    prediction = model.predict(X)
    
    # Also get probability scores if available
    prediction_proba = None
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(X)
    
    return {
        "prediction": int(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
        "probabilities": prediction_proba[0].tolist() if prediction_proba is not None else None
    }


def get_available_models(model_path="Addiction Level_models.pkl"):
    """Get list of available model names."""
    try:
        models = load_models(model_path)
        return list(models.keys())
    except FileNotFoundError:
        return []
