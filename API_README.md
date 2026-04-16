# API Quick Reference

REST API for predicting social media addiction levels.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
./run_api.sh
```

**Access the API:**

- Base URL: [http://localhost:8000](http://localhost:8000)
- Interactive Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpoints


| Method | Endpoint   | Description           |
| ------ | ---------- | --------------------- |
| GET    | `/`        | API information       |
| GET    | `/models`  | List available models |
| POST   | `/predict` | Make prediction       |


## Making a Prediction

**Request:** POST `/predict`

```json
{
  "Age": 25,
  "Gender": "Male",
  "Platform": "Instagram",
  "Total Time Spent": 120,
  "Self Control": 6,
  "model_name": "RandomForest"
  // ... see full example in /docs
}
```

**Response:**

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

## Required Input Fields

All fields are required (except `model_name`). Key fields include:

- **Demographics**: Age, Gender, Location, Income, Profession
- **Platform Usage**: Platform, Total Time Spent, Number of Sessions
- **Video Metrics**: Video Category, Length, Engagement, Videos Watched
- **Behavior**: Self Control, Satisfaction, Watch Reason, Frequency
- **Device**: DeviceType, OS, ConnectionType

## Addiction Level Scale

- **0**: No addiction - Healthy social media usage
- **1**: Very low addiction - Minimal risk
- **2**: Low addiction - Some concerning patterns
- **3**: Moderate addiction - Notable usage concerns
- **4**: High addiction - Significant usage issues
- **5**: Very high addiction - Severe dependency

### Test Script

```bash
python test_api.py  
```

## Available Models

Choose a model using the `model_name` field:

- `RandomForest` (default, recommended)
- `LogisticRegression`
- `KNN`
- `DecisionTree`
- `GradientBoosting`
- `SVM`
- `NaiveBayes`

## Error Responses

- **200** - Success
- **400** - Validation error (check input format)
- **500** - Server error (check logs)

---

