from fastapi import FastAPI, HTTPException, Response, status, Query
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Tuple, Any # Added Tuple and Any
import joblib
import pandas as pd
import os
import uvicorn
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# --- Pydantic Model for your actual features (this remains the same) ---
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

    class Config: # Your example data is fine here
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

# --- NEW Pydantic Model to represent Snowflake's request body structure ---
class SnowflakeRequest(BaseModel):
    data: List[Tuple[int, CarFeatures]]
    # This expects a JSON like: {"data": [[0, {"MILEAGE": ..., "VEHICLE_AGE": ...S}], ...]}

# --- Feature Order (Must match training) ---
# (Your feature_names_in_order remains the same)
numeric_features = ["MILEAGE", "VEHICLE_AGE", "REPORTED_ISSUES", "ACCIDENT_HISTORY", "ENGINE_SIZE", "FUEL_EFFICIENCY", "INSURANCE_PREMIUM", "ODOMETER_READING", "SERVICE_HISTORY"]
categorical_features = ["VEHICLE_MODEL", "MAINTENANCE_HISTORY", "OWNER_TYPE", "FUEL_TYPE", "TRANSMISSION_TYPE", "BATTERY_STATUS", "BRAKE_CONDITION", "TIRE_CONDITION"]
feature_names_in_order = numeric_features + categorical_features


# --- FastAPI App Initialization (remains the same) ---
app = FastAPI(
    title="Car Repair Show/No-Show Predictor (Components)",
    description="API using separate preprocessor and classifier. Snowflake compatible.",
    version="1.4.0" 
)

# --- Model Component Loading (remains the same) ---
# (@app.on_event("startup") def load_components(): ... )
PREPROCESSOR_FILE = "preprocessor_resaved.joblib.gz"
CLASSIFIER_FILE = "classifier_resaved.joblib.gz"
preprocessor: Optional[ColumnTransformer] = None 
classifier: Optional[XGBClassifier] = None      

@app.on_event("startup")
def load_components():
    global preprocessor, classifier
    # ... (your existing loading logic) ...
    # (Make sure your print statements for loading are still there for debugging startup)
    # Load Preprocessor
    if not os.path.exists(PREPROCESSOR_FILE):
        print(f"FATAL ERROR: Preprocessor file not found at '{os.path.abspath(PREPROCESSOR_FILE)}'")
    else:
        try:
            preprocessor = joblib.load(PREPROCESSOR_FILE)
            print(f"Preprocessor loaded successfully from '{PREPROCESSOR_FILE}'.")
        except Exception as e:
            print(f"FATAL ERROR: Failed to load preprocessor from '{PREPROCESSOR_FILE}': {e}")
            preprocessor = None # Ensure it's None if load fails
    # Load Classifier
    if not os.path.exists(CLASSIFIER_FILE):
        print(f"FATAL ERROR: Classifier file not found at '{os.path.abspath(CLASSIFIER_FILE)}'")
    else:
        try:
            classifier = joblib.load(CLASSIFIER_FILE)
            print(f"Classifier loaded successfully from '{CLASSIFIER_FILE}'.")
        except Exception as e:
            print(f"FATAL ERROR: Failed to load classifier from '{CLASSIFIER_FILE}': {e}")
            classifier = None # Ensure it's None if load fails

    if preprocessor is None or classifier is None:
        print("WARNING: One or both model components failed to load initially. Predict endpoint will fail.")


# --- Health Check Endpoints (remain the same) ---
# (@app.get("/", ...) and @app.get("/health", ...))
@app.get("/", tags=["Health Check"])
async def read_root():
    preprocessor_status = "loaded" if preprocessor is not None else "not loaded"
    classifier_status = "loaded" if classifier is not None else "not loaded"
    return {
        "status": "API is running",
        "preprocessor_status": preprocessor_status,
        "classifier_status": classifier_status
    }

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


# --- Prediction Endpoint MODIFIED ---
@app.post("/v1/predict-snowflake", tags=["Prediction"]) # Assuming this is your target path
async def predict(
    # MODIFIED: Change the payload type to expect Snowflake's wrapper
    snowflake_payload: SnowflakeRequest,
    # Query parameter snowflake_format is still useful if you want this endpoint
    # to also serve other clients with a simpler response.
    # API Gateway is already adding this, so it will be true for Snowflake calls.
    snowflake_format_response: bool = Query(True, alias="snowflake_format", description="Return response in Snowflake-compatible format.")
):
    global preprocessor, classifier
    if preprocessor is None or classifier is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Model components are not loaded. Please check server logs."
        )
    try:
        # Extract the actual features for the current row from Snowflake's input format.
        # Snowflake sends one row at a time to this API via API Gateway for each
        # call to the external function in your SQL.
        # So, snowflake_payload.data will be a list containing one element: [0, CarFeatures_object]
        if not snowflake_payload.data or not isinstance(snowflake_payload.data, list) or len(snowflake_payload.data) == 0:
            raise HTTPException(status_code=400, detail="Invalid input: 'data' array is missing or empty.")
        
        # The actual features are in the second element of the first item in the "data" list
        # The type hint SnowflakeRequest.data: List[Tuple[int, CarFeatures]]
        # ensures Pydantic has already validated this structure.
        actual_features: CarFeatures = snowflake_payload.data[0][1]
        
        features_dict = actual_features.model_dump() # Or .dict() for Pydantic v1
        input_df = pd.DataFrame([features_dict], columns=feature_names_in_order)
        
        print("Applying preprocessor...")
        processed_features = preprocessor.transform(input_df)
        print("Preprocessing complete.")
        print("Applying classifier...")
        prediction_numeric = classifier.predict(processed_features)
        print(f"Raw prediction: {prediction_numeric}")
        
        label_map_inv = {1: 'Show', 0: 'No-Show'}
        prediction_label = label_map_inv.get(prediction_numeric[0], 'Unknown Prediction Value')
        
        # Return in appropriate format
        if snowflake_format_response: # This will be true when called from Snowflake via your API GW setup
            # Snowflake-compatible format
            row_index = snowflake_payload.data[0][0] # Get the row index sent by Snowflake
            return {
                "data": [
                    [row_index, prediction_label] 
                ]
            }
        else:
            # Standard API format for other clients (e.g., direct Postman test without snowflake_format=true)
            return {"prediction": prediction_label}
            
    except KeyError as ke:
        print(f"Prediction Error: Missing feature - {ke}")
        raise HTTPException(status_code=400, detail=f"Missing feature in input data: {ke}")
    except ValueError as ve:
        print(f"Prediction Error: Value error - {ve}")
        raise HTTPException(status_code=400, detail=f"Error processing feature values: {ve}")
    except Exception as e:
        print(f"Prediction Error: Unexpected error - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction: {type(e).__name__}"
        )

# --- Optional: Main block to run Uvicorn directly (remains the same) ---
if __name__ == "__main__":
    # ... (your uvicorn.run call) ...
    print("Starting Uvicorn server directly...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)