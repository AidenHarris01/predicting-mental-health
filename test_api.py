import requests
import json


API_URL = "http://localhost:8000/predict"


example_data = {
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
    "ConnectionType": "Wi-Fi",
    "model_name": "DecisionTree"
}

def test_models():
    response = requests.get("http://localhost:8000/models")
    print("Available Models:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
        print()
        print(f"Predicted Addiction Level: {result['addiction_level']}")
        print(f"Description: {result['prediction_details']['addiction_level_description']}")
        if result['probabilities']:
            print(f"Confidence: {result['prediction_details']['confidence']:.2%}")
    else:
        print(f"error: {response.status_code}")
        print(response.json())

if __name__ == "__main__":    
    
    test_models()
    test_predict(example_data)
    high_addiction_data = example_data.copy()
    high_addiction_data.update({
        "Total Time Spent": 300,
        "Number of Sessions": 25,
        "Self Control": 2,
        "Satisfaction": 3,
        "Watch Reason": "Procrastination"
    })
    test_predict(high_addiction_data)
    low_addiction_data = example_data.copy()
    low_addiction_data.update({
        "Total Time Spent": 30,
        "Number of Sessions": 3,
        "Self Control": 9,
        "Satisfaction": 8,
        "Watch Reason": "Entertainment"
    })
    test_predict(low_addiction_data)
        

