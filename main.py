from fastapi import FastAPI, HTTPException, Response, status, Query
from pydantic import BaseModel, Field 
from typing import List, Union, Optional, Tuple, Any 
import joblib
import pandas as pd
import os
import uvicorn
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import boto3 # For S3 interaction
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import tempfile # For temporary file handling

# --- Configuration for S3 Model Loading ---
S3_BUCKET_NAME = 'noshow-model' # Your S3 bucket name
S3_PREPROCESSOR_KEY = 'models/preprocessor_retrained.joblib.gz' # Key for the preprocessor in S3
S3_CLASSIFIER_KEY = 'models/classifier_retrained.joblib.gz'   # Key for the classifier in S3

# --- Pydantic Model for your actual features ---
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
        # Pydantic V2 uses model_config = {"json_schema_extra": {...}}
        # Pydantic V1 uses schema_extra = {...}
        # For compatibility, you might need to adjust based on your Pydantic version
        # If using Pydantic V2:
        # model_config = {
        #     "json_schema_extra": {
        #         "example": {
        #             "MILEAGE": 50000.5, "VEHICLE_AGE": 5, "REPORTED_ISSUES": 1,
        #             # ... rest of your example
        #         }
        #     }
        # }
        # For Pydantic V1 (as originally in your code):
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


# --- Pydantic Model to represent Snowflake's request body structure ---
class SnowflakeRequest(BaseModel):
    data: List[Tuple[int, CarFeatures]]

# --- Feature Order (Must match training) ---
numeric_features = ["MILEAGE", "VEHICLE_AGE", "REPORTED_ISSUES", "ACCIDENT_HISTORY", "ENGINE_SIZE", "FUEL_EFFICIENCY", "INSURANCE_PREMIUM", "ODOMETER_READING", "SERVICE_HISTORY"]
categorical_features = ["VEHICLE_MODEL", "MAINTENANCE_HISTORY", "OWNER_TYPE", "FUEL_TYPE", "TRANSMISSION_TYPE", "BATTERY_STATUS", "BRAKE_CONDITION", "TIRE_CONDITION"]
feature_names_in_order = numeric_features + categorical_features

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Car Repair Show/No-Show Predictor (S3 Models)",
    description="API using separate preprocessor and classifier loaded from S3. Snowflake compatible.",
    version="1.5.0" 
)

# --- Global variables for loaded model components ---
preprocessor: Optional[ColumnTransformer] = None 
classifier: Optional[XGBClassifier] = None     

@app.on_event("startup")
def load_components():
    """Load the preprocessor and classifier from S3 when the application starts."""
    global preprocessor, classifier
    
    # Use a temporary directory to download S3 files
    # This ensures files are cleaned up after loading or if an error occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        local_s3_preprocessor_file = os.path.join(tmpdir, "preprocessor_from_s3.joblib.gz")
        local_s3_classifier_file = os.path.join(tmpdir, "classifier_from_s3.joblib.gz")
        
        preprocessor_loaded_s3 = False
        classifier_loaded_s3 = False
        s3_client = None # Initialize s3_client

        try:
            # Initialize S3 client. 
            # For ECS, this will use the Task Role's credentials.
            # For local testing, it uses your local AWS CLI/env var credentials.
            s3_client = boto3.client('s3') 
            print(f"Attempting to download preprocessor from s3://{S3_BUCKET_NAME}/{S3_PREPROCESSOR_KEY} to {local_s3_preprocessor_file}")
            s3_client.download_file(S3_BUCKET_NAME, S3_PREPROCESSOR_KEY, local_s3_preprocessor_file)
            preprocessor = joblib.load(local_s3_preprocessor_file)
            print(f"Preprocessor loaded successfully from S3 object: s3://{S3_BUCKET_NAME}/{S3_PREPROCESSOR_KEY}")
            preprocessor_loaded_s3 = True

        except NoCredentialsError:
            print(f"S3 Load Error (Preprocessor): AWS credentials not found. Ensure the ECS Task Role (or local environment) has S3 read permissions.")
        except PartialCredentialsError:
            print(f"S3 Load Error (Preprocessor): Incomplete AWS credentials found.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"S3 Load Error (Preprocessor): Object not found at s3://{S3_BUCKET_NAME}/{S3_PREPROCESSOR_KEY}")
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                print(f"S3 Load Error (Preprocessor): Bucket '{S3_BUCKET_NAME}' does not exist.")
            elif e.response['Error']['Code'] == '403': # More specific check for Forbidden
                 print(f"S3 Load Error (Preprocessor): Access Denied (403 Forbidden) when trying to access s3://{S3_BUCKET_NAME}/{S3_PREPROCESSOR_KEY}. Check IAM permissions.")
            else:
                print(f"S3 Load Error (Preprocessor - ClientError): {e}")
        except Exception as e:
            print(f"FATAL ERROR: Failed to load preprocessor from S3: {type(e).__name__} - {e}")

        # Attempt to load classifier only if preprocessor loaded (or always attempt, depending on desired behavior)
        # For robustness, we'll attempt to load it regardless, but it might also fail if credentials are the issue.
        try:
            if not s3_client: # Initialize if previous try block failed before s3_client was set
                s3_client = boto3.client('s3')
            print(f"Attempting to download classifier from s3://{S3_BUCKET_NAME}/{S3_CLASSIFIER_KEY} to {local_s3_classifier_file}")
            s3_client.download_file(S3_BUCKET_NAME, S3_CLASSIFIER_KEY, local_s3_classifier_file)
            classifier = joblib.load(local_s3_classifier_file)
            print(f"Classifier loaded successfully from S3 object: s3://{S3_BUCKET_NAME}/{S3_CLASSIFIER_KEY}")
            classifier_loaded_s3 = True

        except NoCredentialsError: # Should have been caught by preprocessor load already if it's a general credential issue
            print(f"S3 Load Error (Classifier): AWS credentials not found.")
        except PartialCredentialsError:
            print(f"S3 Load Error (Classifier): Incomplete AWS credentials found.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"S3 Load Error (Classifier): Object not found at s3://{S3_BUCKET_NAME}/{S3_CLASSIFIER_KEY}")
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                 print(f"S3 Load Error (Classifier): Bucket '{S3_BUCKET_NAME}' does not exist.")
            elif e.response['Error']['Code'] == '403': # More specific check for Forbidden
                 print(f"S3 Load Error (Classifier): Access Denied (403 Forbidden) when trying to access s3://{S3_BUCKET_NAME}/{S3_CLASSIFIER_KEY}. Check IAM permissions.")
            else:
                print(f"S3 Load Error (Classifier - ClientError): {e}")
        except Exception as e:
            print(f"FATAL ERROR: Failed to load classifier from S3: {type(e).__name__} - {e}")

        if not preprocessor_loaded_s3 or not classifier_loaded_s3:
            print("WARNING: One or both model components failed to load from S3. Predict endpoint will likely fail.")
            # Ensure they are None if loading failed to prevent using partially loaded models
            if not preprocessor_loaded_s3: preprocessor = None
            if not classifier_loaded_s3: classifier = None
        # Temporary files (local_s3_preprocessor_file, local_s3_classifier_file)
        # are automatically cleaned up when the 'with tempfile.TemporaryDirectory() as tmpdir:' block exits.

# --- Health Check Endpoints ---
@app.get("/", tags=["Health Check"])
async def read_root():
    preprocessor_status = "loaded from S3" if preprocessor is not None else "not loaded"
    classifier_status = "loaded from S3" if classifier is not None else "not loaded"
    return {
        "status": "API is running",
        "preprocessor_status": preprocessor_status,
        "classifier_status": classifier_status
    }

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

# --- Prediction Endpoint ---
# This endpoint logic remains the same as it uses the global preprocessor and classifier
@app.post("/v1/predict-snowflake", tags=["Prediction"])
async def predict(
    snowflake_payload: SnowflakeRequest,
    snowflake_format_response: bool = Query(True, alias="snowflake_format", description="Return response in Snowflake-compatible format.")
):
    global preprocessor, classifier
    if preprocessor is None or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model components are not loaded (failed to load from S3). Please check server logs."
        )
    
    response_data_for_snowflake = []

    try:
        if not snowflake_payload.data or not isinstance(snowflake_payload.data, list):
            raise HTTPException(status_code=400, detail="Invalid input: 'data' array is missing or empty in payload.")

        for item in snowflake_payload.data:
            if not isinstance(item, tuple) or len(item) != 2:
                print(f"Skipping malformed item in batch: {item}")
                continue 

            row_index: int = item[0]
            actual_features: CarFeatures = item[1]
            current_prediction_label = "Error: Prediction failed for this row" 

            try:
                features_dict = actual_features.model_dump() # Use model_dump() for Pydantic V2+ or .dict() for V1
                input_df = pd.DataFrame([features_dict], columns=feature_names_in_order)
                                
                processed_features = preprocessor.transform(input_df)
                prediction_numeric = classifier.predict(processed_features)
                
                label_map_inv = {1: 'Show', 0: 'No-Show'}
                current_prediction_label = label_map_inv.get(prediction_numeric[0], 'Error: Unknown prediction value')
            
            except Exception as e_inner:
                print(f"Error processing features for row index {row_index}: {type(e_inner).__name__} - {e_inner}")
            
            response_data_for_snowflake.append([row_index, current_prediction_label])
        
        if snowflake_format_response:
            return {"data": response_data_for_snowflake}
        else:
            if len(response_data_for_snowflake) == 1:
                return {"prediction": response_data_for_snowflake[0][1]}
            else: 
                return {"predictions_batch": response_data_for_snowflake} 
            
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

# --- Optional: Main block to run Uvicorn directly ---
if __name__ == "__main__":
    print("Starting Uvicorn server directly...")
    # This assumes your FastAPI app instance is named 'app' in a file named 'main.py'
    # If your file is named differently, adjust "main:app" accordingly.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
