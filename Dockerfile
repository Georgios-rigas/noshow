# Dockerfile for FastAPI application loading models from S3

# 1. Use an official Python runtime as a parent image
# Choose a version compatible with your dependencies (e.g., 3.9 or 3.10)
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
# Explicitly copy to the WORKDIR (/app)
COPY requirements.txt /app/requirements.txt

# 4. Install any needed system dependencies (if any)
# Example: RUN apt-get update && apt-get install -y --no-install-recommends <some-package> && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# This will install boto3 if it's in your requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. Copy the application code
# Explicitly copy to the WORKDIR (/app)
# This should include your main.py (or equivalent FastAPI application file)
COPY main.py /app/main.py
# If you have other Python modules or subdirectories for your app, copy them too:
# Example: COPY ./app_code_folder /app/app_code_folder

# --- DEBUG STEP (Optional, can be removed for production) ---
# List the contents of /app to verify what's been copied
RUN echo "Contents of /app after copying application code:" && ls -la /app
# --- END DEBUG STEP ---

# 7. Model component files are NO LONGER COPIED here.
#    The FastAPI application will download them from S3 at startup.
#
# REMOVED: COPY preprocessor_resaved.joblib.gz /app/preprocessor_resaved.joblib.gz
# REMOVED: COPY classifier_resaved.joblib.gz /app/classifier_resaved.joblib.gz

# 8. Expose the port the app runs on (Uvicorn will bind to this port)
EXPOSE 8000

# 9. Define the command to run the application using Uvicorn
# Ensure your main FastAPI app instance is named 'app' in 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
