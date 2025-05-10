from fastapi import FastAPI, HTTPException, Response, status, Query
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import joblib
import pandas as pd
import os
import uvicorn
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# --- Pydantic Model for Input Validation ---
class CarFeatures(BaseModel):
    MILEAGE: float
    VEHICLE_AGE: int
    REPORTED_ISSUES: int
    ACCIDENT_HISTORY: int
    ENGINE_SIZE: float
    FUEL_EFFICIENCY: float
    INSURANCE_PREMIUM: float
    ODOMETER_READING: float
    SERVICE_HISTORY: int
    VEHICLE_MODEL: str
    MAINTENANCE_HISTORY: str
    OWNER_TYPE: str
    FUEL_TYPE: str
    TRANSMISSION_TYPE: str
    BATTERY_STATUS: str
    BRAKE_CONDITION: str
    TIRE_CONDITION: str

    class Config:
        schema_extra = {
            "example": {
                "MILEAGE": 50000.5, "VEHICLE_AGE": 5, "REPORTED_ISSUES": 1,
                "ACCIDENT_HISTORY": 0, "ENGINE_SIZE": 1.6, "FUEL_EFFICIENCY": 30.5,
                "INSURANCE_PREMIUM": 500.0, "ODOMETER_READING": 50000.5, "SERVICE_HISTORY": 3,
                "VEHICLE_MODEL": "Sedan", "MAINTENANCE_HISTORY": "Regular", "OWNER_TYPE": "First",
                "FUEL_TYPE": "Gasoline", "TRANSMISSION_TYPE": "Automatic", "BATTERY_STATUS": "Good",
                "BRAKE_CONDITION": "Fair", "TIRE_CONDITION": "Good"
            }
        }

class PredictionInput(BaseModel):
    features: CarFeatures

# --- Feature Order (Must match training) ---
numeric_features = ["MILEAGE", "VEHICLE_AGE", "REPORTED_ISSUES", "ACCIDENT_HISTORY", "ENGINE_SIZE", "FUEL_EFFICIENCY", "INSURANCE_PREMIUM", "ODOMETER_READING", "SERVICE_HISTORY"]
categorical_features = ["VEHICLE_MODEL", "MAINTENANCE_HISTORY", "OWNER_TYPE", "FUEL_TYPE", "TRANSMISSION_TYPE", "BATTERY_STATUS", "BRAKE_CONDITION", "TIRE_CONDITION"]
feature_names_in_order = numeric_features + categorical_features

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Car Repair Show/No-Show Predictor (Components)",
    description="API using separate preprocessor and classifier.",
    version="1.2.0"  # Incremented version to reflect Snowflake compatibility
)

# --- Model Component Loading ---
PREPROCESSOR_FILE = "preprocessor_resaved.joblib.gz"
CLASSIFIER_FILE = "classifier_resaved.joblib.gz"
preprocessor: ColumnTransformer = None
classifier: XGBClassifier = None

@app.on_event("startup")
def load_components():
    """Load the preprocessor and classifier when the application starts."""
    global preprocessor, classifier
    preprocessor_loaded = False
    classifier_loaded = False
    # Load Preprocessor
    if not os.path.exists(PREPROCESSOR_FILE):
        print(f"FATAL ERROR: Preprocessor file not found at '{os.path.abspath(PREPROCESSOR_FILE)}'")
    else:
        try:
            preprocessor = joblib.load(PREPROCESSOR_FILE)
            print(f"Preprocessor loaded successfully from '{PREPROCESSOR_FILE}'.")
            preprocessor_loaded = True
        except Exception as e:
            print(f"FATAL ERROR: Failed to load preprocessor from '{PREPROCESSOR_FILE}': {e}")
    # Load Classifier
    if not os.path.exists(CLASSIFIER_FILE):
        print(f"FATAL ERROR: Classifier file not found at '{os.path.abspath(CLASSIFIER_FILE)}'")
    else:
        try:
            classifier = joblib.load(CLASSIFIER_FILE)
            print(f"Classifier loaded successfully from '{CLASSIFIER_FILE}'.")
            classifier_loaded = True
        except Exception as e:
            print(f"FATAL ERROR: Failed to load classifier from '{CLASSIFIER_FILE}': {e}")
    if not preprocessor_loaded or not classifier_loaded:
        print("WARNING: One or both model components failed to load. Predict endpoint will fail.")
        preprocessor = None
        classifier = None

# --- Health Check Endpoint ---
@app.get("/", tags=["Health Check"])
async def read_root():
    """Basic health check endpoint."""
    preprocessor_status = "loaded" if preprocessor is not None else "not loaded"
    classifier_status = "loaded" if classifier is not None else "not loaded"
    return {
        "status": "API is running",
        "preprocessor_status": preprocessor_status,
        "classifier_status": classifier_status
    }

# --- Dedicated Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health Check"])
async def health_check():
    """Simple health check endpoint that returns status 200 OK."""
    # This endpoint does nothing complex, just returns OK.
    # Useful for Load Balancer health checks.
    return {"status": "ok"}

# --- Prediction Endpoint with Snowflake Compatibility ---
@app.post("/predict", tags=["Prediction"])
async def predict(payload: PredictionInput, snowflake_format: bool = Query(False, description="Return response in Snowflake-compatible format")):
    """Predicts show/no-show using separate preprocessor and classifier.
    
    Set snowflake_format=true for Snowflake External Function compatibility.
    """
    global preprocessor, classifier
    if preprocessor is None or classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model components are not loaded. Please check server logs."
        )
    try:
        # Extract features
        features_dict = payload.features.dict()
        input_df = pd.DataFrame([features_dict], columns=feature_names_in_order)
        
        # Debug prints
        print("Applying preprocessor...")
        processed_features = preprocessor.transform(input_df)
        print("Preprocessing complete.")
        print("Applying classifier...")
        prediction_numeric = classifier.predict(processed_features)
        print(f"Raw prediction: {prediction_numeric}")
        
        # Convert numeric prediction to label
        label_map_inv = {1: 'Show', 0: 'No-Show'}
        prediction_label = label_map_inv.get(prediction_numeric[0], 'Unknown Prediction Value')
        
        # Return in appropriate format
        if snowflake_format:
            # Snowflake-compatible format
            return {
                "data": [
                    [prediction_label]
                ]
            }
        else:
            # Standard API format for other clients
            return {"prediction": prediction_label}
            
    except KeyError as ke:
        print(f"Prediction Error: Missing feature - {ke}")
        raise HTTPException(status_code=400, detail=f"Missing feature in input data: {ke}")
    except ValueError as ve:
        print(f"Prediction Error: Value error - {ve}")
        raise HTTPException(status_code=400, detail=f"Error processing feature values: {ve}")
    except Exception as e:
        print(f"Prediction Error: Unexpected error - {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction."
        )

# --- Optional: Main block to run Uvicorn directly ---
if __name__ == "__main__":
    print("Starting Uvicorn server directly...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



