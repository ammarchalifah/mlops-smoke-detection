# libraries
import os
import logging

import mlflow
import pandas as pd
import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient


# Env Vars
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS")
EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE_ADDRESS")

MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "smoke_detection")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# Flask Applications
app = Flask("Smoke Detection")

# Mongo
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


# Load Model
logged_model = f"models:/{MLFLOW_MODEL_NAME}/Production"
model = mlflow.pyfunc.load_model(logged_model)


def preprocess(input_dict):
    """
    Preprocess incoming dataset
    """
    data = pd.DataFrame.from_dict(input_dict, orient="index").T
    return data


def predict(features):
    """
    Predict function
    """
    preds = model.predict(features)
    return float(preds[0])


def save_to_db(record, prediction):
    rec = record.copy()
    rec["prediction"] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/smoke_detection", json=[rec])
    logging.info(f"Logged data to evidently row:{str(rec)}")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    input_json = request.get_json()
    features = preprocess(input_json)
    pred = predict(features)

    result = {"prediction": pred}

    # Log results to Evidently & Mongo
    logging.info("Saving data to MongoDB and Evidently services")
    save_to_db(input_json, pred)
    send_to_evidently_service(input_json, pred)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
