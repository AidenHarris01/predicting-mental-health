# Social Media Addiction Prediction System

A machine learning API that predicts social media addiction levels (0-5) based on user behavior, demographics, and platform usage patterns.

**Tech Stack:** Python • FastAPI • scikit-learn • Pydantic

---

## Quick Start

### 1. Install Dependencies

```bash

python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

### 2. Train Models (if needed)

```bash
python src/train.py
```

This trains 7 ML models and saves them to `Addiction Level_models.pkl`.

### 3. Start the API Server

```bash
./run_api.sh
```

The API will be available at:

- **API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Making Predictions With Test File

```bash
python test_api.py
```

---

## How It Works

### System Architecture (in assets folder)

The system follows a 4-layer architecture:

1. **Client** - Sends requests via HTTP
2. **FastAPI Server** - Validates input with Pydantic
3. **Preprocessing** - One-hot encoding, feature alignment, scaling
4. **Prediction** - Loads ML model and generates predictions

### Prediction Flow

```
User Input → Validation → Preprocessing → Model → Prediction + Confidence
```

---

## Available Models

Choose a model using the `model_name` field:

- **RandomForest** (default, recommended)
- LogisticRegression
- KNN
- DecisionTree
- GradientBoosting
- SVM
- NaiveBayes

---

## API Endpoints


| Method | Endpoint   | Description                                   |
| ------ | ---------- | --------------------------------------------- |
| GET    | `/`        | API information                               |
| GET    | `/models`  | List available models                         |
| POST   | `/predict` | Make prediction (returns addiction level 0-5) |


### Prediction Response

```json
{
  "addiction_level": 3,
  "probabilities": [0.05, 0.10, 0.15, 0.40, 0.20, 0.10],
  "model_used": "RandomForest",
  "prediction_details": {
    "addiction_level_description": "Moderate addiction - Notable usage concerns",
    "confidence": 0.40
  }
}
```

---

## Required Input Fields

All fields are required (except `model_name`). Key categories:

- **Demographics**: Age, Gender, Location, Income, Profession
- **Platform Usage**: Platform, Total Time Spent, Sessions
- **Video Metrics**: Category, Length, Engagement, Videos Watched
- **Behavior**: Self Control, Satisfaction, Watch Reason, Frequency
- **Device**: DeviceType, OS, ConnectionType

---

## Model Training & Evaluation

### Train Models

```bash
python src/train.py
```

- Trains 7 models with hyperparameter tuning (GridSearchCV)
- Uses stratified 70/15/15 train/val/test split
- Saves best models to pickle file

### Evaluate Models

```bash
python src/evaluate.py
```

Generates classification reports and confusion matrices for all models.

### Exploratory Data Analysis

```bash
jupyter notebook exploratory_analysis.ipynb
```



---

## Dataset

**Source:** Time Wasters on Social Media  
**Target:** Addiction Level (0-5)



