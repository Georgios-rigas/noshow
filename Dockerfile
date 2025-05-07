    # Dockerfile

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
RUN pip install --no-cache-dir --upgrade pip
# Use the copied requirements file path explicitly
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. Copy the application code
# Explicitly copy to the WORKDIR (/app)
COPY main.py /app/main.py

# --- DEBUG STEP ---
# List the contents of /app before copying model files
RUN echo "Contents of /app before copying models:" && ls -la /app
# --- END DEBUG STEP ---

# 7. Copy the model component files
# Explicitly copy to the WORKDIR (/app)
COPY preprocessor_resaved.joblib.gz /app/preprocessor_resaved.joblib.gz
COPY classifier_resaved.joblib.gz /app/classifier_resaved.joblib.gz

# 8. Expose the port the app runs on
EXPOSE 8000

# 9. Define the command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]