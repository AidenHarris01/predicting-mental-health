import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from sklearn.preprocessing import StandardScaler

def load_models(model_path: str = "Addiction Level_models.pkl"):
    """
    Load all trained models from pickle file
    
    Args:
        model_path (str): Path to the pickle file containing the models dictionary
        
    Returns:
        dict: Dictionary of trained machine learning models
    """
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models

def load_model(model_name: str = "RandomForest", model_path: str = "Addiction Level_models.pkl"):
    """
    Load a specific trained model from pickle file
    
    Args:
        model_name (str): Name of the model to load (e.g., 'RandomForest', 'LogisticRegression', 'KNN', 'DecisionTree', 'GradientBoosting', 'SVM', 'NaiveBayes')
        model_path (str): Path to the pickle file containing the models dictionary
        
    Returns:
        model: The trained machine learning model
    """
    models = load_models(model_path)
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    return models[model_name]

def preprocess_input(input_data: Dict[str, Any], training_columns: list = None, scale_features: bool = True) -> pd.DataFrame:
    """
    Preprocess input data to match the format expected by the model
    
    Args:
        input_data (dict): Dictionary containing the input features
        training_columns (list): List of feature column names from training
        scale_features (bool): Whether to scale features (should match training)
        
    Returns:
        pd.DataFrame: Preprocessed features ready for prediction
    """
    # If training columns not provided, we need to infer them from the dataset
    if training_columns is None:
        # Load a small sample from the dataset to get all possible columns
        df_sample = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")
        # Remove target columns
        df_sample = df_sample.drop(["Addiction Level", "ProductivityLoss"], axis=1)
        # One-hot encode to get all possible columns
        df_sample = pd.get_dummies(df_sample, drop_first=True)
        training_columns = df_sample.columns.tolist()
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Handle missing target columns if they exist
    if "Addiction Level" in df.columns:
        df = df.drop(["Addiction Level"], axis=1)
    if "ProductivityLoss" in df.columns:
        df = df.drop(["ProductivityLoss"], axis=1)
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    
    # Align with training columns - add missing columns with 0s
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Remove extra columns and ensure same order as training
    df = df[training_columns]
    
    # Scale if required (note: this is a simplified scaling, ideally we'd use the fitted scaler)
    if scale_features:
        scaler = StandardScaler()
        # Load sample data to fit scaler properly
        df_full = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")
        df_full = df_full.drop(["Addiction Level", "ProductivityLoss"], axis=1)
        df_full = pd.get_dummies(df_full, drop_first=True)
        df_full = df_full[training_columns]
        scaler.fit(df_full)
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df

def predict(input_data: Union[Dict[str, Any], pd.DataFrame], model_name: str = "RandomForest", model_path: str = "Addiction Level_models.pkl", scale_features: bool = True):
    """
    Predict addiction level using the trained model
    
    Args:
        input_data: Either a dictionary with feature values or a preprocessed DataFrame
        model_name (str): Name of the model to use for prediction (default: 'RandomForest')
        model_path (str): Path to the pickle file containing the models
        scale_features (bool): Whether to scale features (should match training, default: True)
        
    Returns:
        prediction: The predicted addiction level
    """
    model = load_model(model_name, model_path)
    
    # Preprocess input if it's a dictionary
    if isinstance(input_data, dict):
        X = preprocess_input(input_data, training_columns=None, scale_features=scale_features)
    else:
        X = input_data
    
    # Make prediction
    prediction = model.predict(X)
    
    return prediction[0] if len(prediction) == 1 else prediction

def predict_proba(input_data: Union[Dict[str, Any], pd.DataFrame], model_name: str = "RandomForest", model_path: str = "Addiction Level_models.pkl", scale_features: bool = True):
    """
    Predict addiction level probabilities using the trained model
    
    Args:
        input_data: Either a dictionary with feature values or a preprocessed DataFrame
        model_name (str): Name of the model to use for prediction (default: 'RandomForest')
        model_path (str): Path to the pickle file containing the models
        scale_features (bool): Whether to scale features (should match training, default: True)
        
    Returns:
        probabilities: Array of probabilities for each class
    """
    model = load_model(model_name, model_path)
    
    # Preprocess input if it's a dictionary
    if isinstance(input_data, dict):
        X = preprocess_input(input_data, training_columns=None, scale_features=scale_features)
    else:
        X = input_data
    
    # Check if model supports probability predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return probabilities[0] if len(probabilities) == 1 else probabilities
    else:
        raise AttributeError("Model does not support probability predictions")

def predict_with_details(input_data: Dict[str, Any], model_name: str = "RandomForest", model_path: str = "Addiction Level_models.pkl", scale_features: bool = True):
    """
    Predict addiction level with detailed output including probabilities
    
    Args:
        input_data (dict): Dictionary containing the input features
        model_name (str): Name of the model to use for prediction (default: 'RandomForest')
        model_path (str): Path to the pickle file containing the models
        scale_features (bool): Whether to scale features (should match training, default: True)
        
    Returns:
        dict: Dictionary containing prediction and probabilities (if available)
    """
    prediction = predict(input_data, model_name, model_path, scale_features)
    
    result = {
        "model": model_name,
        "prediction": int(prediction),
    }
    
    try:
        probabilities = predict_proba(input_data, model_name, model_path, scale_features)
        result["probabilities"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
        result["confidence"] = float(max(probabilities))
    except (AttributeError, Exception):
        pass
    
    return result

if __name__ == "__main__":
    # Example usage
    example_input = {
        "Age": 25,
        "Gender": "Male",
        "Location": "United States",
        "Income": 50000,
        "Debt": False,
        "Owns Property": False,
        "Profession": "Engineer",
        "Demographics": "Urban",
        "Platform": "Instagram",
        "Total Time Spent": 120,
        "Number of Sessions": 15,
        "Video ID": 1234,
        "Video Category": "Entertainment",
        "Video Length": 10,
        "Engagement": 8000,
        "Importance Score": 5,
        "Time Spent On Video": 30,
        "Number of Videos Watched": 20,
        "Scroll Rate": 70,
        "Frequency": "Evening",
        "Satisfaction": 6,
        "Watch Reason": "Entertainment",
        "DeviceType": "Smartphone",
        "OS": "iOS",
        "Watch Time": "7:00 PM",
        "Self Control": 6,
        "CurrentActivity": "At home",
        "ConnectionType": "Wi-Fi"
    }
    
    try:
        # Make prediction with default RandomForest model
        print("=== Using RandomForest Model ===")
        prediction = predict(example_input)
        print(f"Predicted Addiction Level: {prediction}")
        
        # Get probability predictions if available
        try:
            probabilities = predict_proba(example_input)
            print(f"Prediction Probabilities: {probabilities}")
        except AttributeError:
            print("Model does not support probability predictions")
        
        # Try with detailed output
        print("\n=== Detailed Prediction ===")
        result = predict_with_details(example_input, model_name="RandomForest")
        print(f"Result: {result}")
        
        # Show available models
        print("\n=== Available Models ===")
        models = load_models()
        print(f"Models: {list(models.keys())}")
            
    except Exception as e:
        print(f"Error: {e}")
