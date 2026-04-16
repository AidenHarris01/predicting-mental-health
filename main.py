import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_models(model_path="Addiction Level_models.pkl"):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_model(model_name="RandomForest", model_path="Addiction Level_models.pkl"):
    models = load_models(model_path)
    return models[model_name]

def preprocess_input(input_data, training_columns=None, scale_features=True):
    if training_columns is None:
        df_sample = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")
        df_sample = df_sample.drop(["Addiction Level", "ProductivityLoss"], axis=1)
        df_sample = pd.get_dummies(df_sample, drop_first=True)
        training_columns = df_sample.columns.tolist()
    
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, drop_first=True)
    
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[training_columns]
    
    if scale_features:
        scaler = StandardScaler()
        df_full = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")
        df_full = df_full.drop(["Addiction Level", "ProductivityLoss"], axis=1)
        df_full = pd.get_dummies(df_full, drop_first=True)
        df_full = df_full[training_columns]
        scaler.fit(df_full)
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df

def predict(input_data, model_name="RandomForest", model_path="Addiction Level_models.pkl", scale_features=True):
    model = load_model(model_name, model_path)
    
    if isinstance(input_data, dict):
        X = preprocess_input(input_data, training_columns=None, scale_features=scale_features)
    else:
        X = input_data
    
    prediction = model.predict(X)
    return prediction[0] if len(prediction) == 1 else prediction


if __name__ == "__main__":
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
    
    
    prediction = predict(example_input)
    print(f"Prediction: {prediction}")