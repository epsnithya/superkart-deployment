FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# xgboost needs the OpenMP runtime, which is missing from slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the serialized model
COPY . .

# Flask API runs on port 7860
EXPOSE 7860

# Serve the API with gunicorn (production-grade WSGI server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
