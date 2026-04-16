import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def load_models(model_path="Addiction Level_models.pkl"):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_model(model_name="DecisionTree", model_path="Addiction Level_models.pkl"):
    models = load_models(model_path)
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    return models[model_name]


def get_training_columns(data_path="data/Time_Wasters_on_Social_Media.csv"):
    df_sample = pd.read_csv(data_path)
    df_sample = df_sample.drop(["Addiction Level", "ProductivityLoss", "Self Control", "Satisfaction"], axis=1)
    df_sample = pd.get_dummies(df_sample, drop_first=True)
    return df_sample.columns.tolist()


def preprocess_input(input_data, training_columns=None, scale_features=True, data_path="data/Time_Wasters_on_Social_Media.csv"):
    if training_columns is None:
        training_columns = get_training_columns(data_path)
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, drop_first=True)
    
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]
    
    if scale_features:
        scaler = StandardScaler()
        df_full = pd.read_csv(data_path)
        df_full = df_full.drop(["Addiction Level", "ProductivityLoss", "Self Control", "Satisfaction"], axis=1)
        df_full = pd.get_dummies(df_full, drop_first=True)
        df_full = df_full[training_columns]
        scaler.fit(df_full)
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df


def predict(input_data, model_name="DecisionTree", model_path="Addiction Level_models.pkl", 
            scale_features=True, data_path="data/Time_Wasters_on_Social_Media.csv"):

    model = load_model(model_name, model_path)
    if isinstance(input_data, dict):
        X = preprocess_input(input_data, training_columns=None, 
                            scale_features=scale_features, data_path=data_path)
    else:
        X = input_data
    
    prediction = model.predict(X)
    prediction_proba = None
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(X)
    
    return {
        "prediction": int(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
        "probabilities": prediction_proba[0].tolist() if prediction_proba is not None else None
    }


def get_available_models(model_path="Addiction Level_models.pkl"):
    try:
        models = load_models(model_path)
        return list(models.keys())
    except FileNotFoundError:
        return []
