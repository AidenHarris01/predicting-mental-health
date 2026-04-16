from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict, get_available_models

app = FastAPI()

class PredictionInput(BaseModel):
    """Input schema"""
    
    # Demographics
    Age: int = Field(..., ge=13, le=100, description="User age (13-100)")
    Gender: Literal["Male", "Female", "Other"] = Field(..., description="User gender")
    Location: str = Field(..., description="User location/country")
    Income: int = Field(..., ge=0, description="Annual income in USD")
    Debt: bool = Field(..., description="Whether user has debt")
    Owns_Property: bool = Field(..., alias="Owns Property", description="Whether user owns property")
    Profession: str = Field(..., description="User profession")
    Demographics: Literal["Urban", "Rural"] = Field(..., description="Urban or Rural area")
    
    # Platform Usage
    Platform: Literal["Instagram", "Facebook", "TikTok", "YouTube"] = Field(..., description="Social media platform")
    Total_Time_Spent: int = Field(..., ge=0, alias="Total Time Spent", description="Total time spent in minutes")
    Number_of_Sessions: int = Field(..., ge=0, alias="Number of Sessions", description="Number of usage sessions")
    
    # Video Information
    Video_ID: int = Field(..., ge=0, alias="Video ID", description="Video identifier")
    Video_Category: str = Field(..., alias="Video Category", description="Category of video content")
    Video_Length: int = Field(..., ge=0, alias="Video Length", description="Length of video in minutes")
    Engagement: int = Field(..., ge=0, description="Engagement score")
    Importance_Score: int = Field(..., ge=1, le=10, alias="Importance Score", description="Importance score (1-10)")
    Time_Spent_On_Video: int = Field(..., ge=0, alias="Time Spent On Video", description="Time spent on video in minutes")
    Number_of_Videos_Watched: int = Field(..., ge=0, alias="Number of Videos Watched", description="Number of videos watched")
    Scroll_Rate: int = Field(..., ge=0, le=100, alias="Scroll Rate", description="Scroll rate percentage (0-100)")
    
    # Usage Patterns
    Frequency: Literal["Morning", "Afternoon", "Evening", "Night"] = Field(..., description="Time of day")
    Satisfaction: int = Field(..., ge=0, le=10, description="Satisfaction level (0-10)")
    Watch_Reason: Literal["Entertainment", "Boredom", "Habit", "Procrastination"] = Field(..., alias="Watch Reason", description="Reason for watching")
    
    # Device Information
    DeviceType: Literal["Smartphone", "Tablet", "Computer"] = Field(..., description="Type of device used")
    OS: Literal["Android", "iOS", "Windows", "MacOS"] = Field(..., description="Operating system")
    Watch_Time: str = Field(..., alias="Watch Time", description="Time of watching (e.g., '7:00 PM')")
    
    # Behavioral Metrics
    Self_Control: int = Field(..., ge=0, le=10, alias="Self Control", description="Self control rating (0-10)")
    CurrentActivity: str = Field(..., description="Current user activity")
    ConnectionType: Literal["Wi-Fi", "Mobile Data"] = Field(..., description="Type of internet connection")
    
    # Model selection
    model_name: Optional[str] = Field("DecisionTree", description="Model to use for prediction")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    addiction_level: int = Field(..., description="Predicted addiction level (0-5)")
    probabilities: Optional[list[float]] = Field(None, description="Probability distribution across all classes")
    model_used: str = Field(..., description="Name of the model used for prediction")
    prediction_details: dict = Field(..., description="Additional prediction details")


@app.get("/")
def read_root():
    return {"test": "test"}


@app.get("/models")
def list_models():
    model_path = os.path.join(os.path.dirname(__file__), "Addiction Level_models.pkl")
    try:
        models = get_available_models(model_path)
        return {"available_models": models,"default_model": "DecisionTree"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_addiction(input_data: PredictionInput):
    try:
        data_dict = input_data.model_dump(by_alias=True)
        model_name = data_dict.pop("model_name", "DecisionTree")
        model_path = os.path.join(os.path.dirname(__file__), "Addiction Level_models.pkl")
        data_path = os.path.join(os.path.dirname(__file__), "data", "Time_Wasters_on_Social_Media.csv")
        result = predict(
            input_data=data_dict,
            model_name=model_name,
            model_path=model_path,
            scale_features=True,
            data_path=data_path
        )
        response = PredictionResponse(
            addiction_level=result["prediction"],
            probabilities=result["probabilities"],
            model_used=model_name,
            prediction_details={
                "addiction_level_description": get_addiction_description(result["prediction"]),
                "confidence": max(result["probabilities"]) if result["probabilities"] else None
            }
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Required file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def get_addiction_description(level: int) -> str:
    """Get human-readable description of addiction level."""
    descriptions = {
        0: "No addiction - Healthy social media usage",
        1: "Very low addiction - Minimal risk",
        2: "Low addiction - Some concerning patterns",
        3: "Moderate addiction - Notable usage concerns",
        4: "High addiction - Significant usage issues",
        5: "Very high addiction - Severe dependency"
    }
    return descriptions.get(level, "Unknown level")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
