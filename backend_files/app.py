# Importing necessary libraries
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained model pipeline (preprocessing + regressor)
model = joblib.load("superkart_model.joblib")

# Features expected by the model
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]


# Home route
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


# Online (single) prediction
@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        data = request.get_json()
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
        prediction = float(model.predict(input_df)[0])
        return jsonify({"Predicted_Sales": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Batch prediction from a CSV file
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        file = request.files["file"]
        input_df = pd.read_csv(file)

        missing = [f for f in FEATURES if f not in input_df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        predictions = model.predict(input_df[FEATURES]).round(2).tolist()
        # {row_index: predicted_sales}
        return jsonify({str(i): p for i, p in zip(input_df.index, predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app locally (inside Docker, gunicorn is used instead)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
