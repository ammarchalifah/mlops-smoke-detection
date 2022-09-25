import os
import scipy as sp
import gc
import pandas as pd
import numpy as np
import mlflow

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

import xgboost as xgb

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

"""
This Prefect flow execute model training using various models and hyperparameter. Select the best performing model from MLflow.

ML problem: classification of smoke presence (binary classification) from data recorded in smoke detection instrument.
The data is from Kaggle competition.

Models to compare:
1. RandomForest
2. XGBoost
3. CatBoost
4. LightGBM
"""


RANDOM_STATE = 1111
FOLDS = 5
TRACKING_SERVER_HOST = "mlflow_web_server"


# Function to read and clean data
@task
def prepare_data(train_path):
    """
    Prepared data for modelling
    """
    df = pd.read_csv(train_path)
    print(df.shape)

    features = [
        "UTC",
        "Temperature",
        "Humidity",
        "TVOC",
        "eCO2",
        "Raw H2",
        "Raw Ethanol",
        "Pressure",
        "PM1.0",
        "PM2.5",
        "NC0.5",
        "NC1.0",
        "NC2.5",
        "CNT",
    ]
    label = "Fire Alarm"

    df_X = df[features]
    df_y = df.loc[:, label]

    return df_X, df_y


# Function to train model without hyperparameter tuning with specified model
@task
def train_model_exploration(df_X, df_y, model):
    """
    Training with model passed with clf argument
    """

    models = {
        "Random Forest": RandomForestClassifier(n_jobs=-1),
        "LightGBM": LGBMClassifier(n_jobs=-1),
        "XGBoost": XGBClassifier(n_jobs=-1),
        "CatBoost": CatBoostClassifier(silent=True),
    }

    with mlflow.start_run():
        mlflow.set_experiment("smoke_detection")
        mlflow.set_tag("Developer", "Ammar Chalifah")
        mlflow.set_tag("Experiment Stage", "Exploration")
        mlflow.log_param("Train Data", "smoke_detection_iot.csv")
        mlflow.log_param("Random State", RANDOM_STATE)
        mlflow.log_param("Folds", FOLDS)
        mlflow.log_param("Models", model)

        gc.collect()

        folds = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

        pred_y = np.zeros(df_X.shape[0])

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_X, df_y)):
            train_X, train_y = df_X.iloc[train_idx], df_y.iloc[train_idx]
            valid_X, valid_y = df_X.iloc[valid_idx], df_y.iloc[valid_idx]

            clf = models[model]

            clf.fit(train_X, train_y)

            pred_y[valid_idx] = clf.predict(valid_X)

            del clf, train_X, train_y, valid_X, valid_y
            gc.collect()

        score = f1_score(df_y, pred_y)
        full_accuracy_score = accuracy_score(df_y, pred_y)

        # Log metric to MLflow
        mlflow.log_metric("f1_score", score)
        mlflow.log_metric("accuracy", full_accuracy_score)


# Function to hyperparameter tuning for XGBoost
def objective_xgboost(params):
    """
    Hypertune XGBoost model (the best model from experiment)
    """
    with mlflow.start_run():
        mlflow.set_experiment("smoke_detection")
        mlflow.set_tag("Developer", "Ammar Chalifah")
        mlflow.set_tag("Experiment Stage", "Hyperparameter Tuning")
        mlflow.log_param("Train Data", "smoke_detection_iot.csv")
        mlflow.log_param("Random State", RANDOM_STATE)
        mlflow.log_param("Folds", FOLDS)
        mlflow.log_param("Models", "XGBoost")
        mlflow.log_params(params)

        gc.collect()

        # TODO: Change this code block to make it modular
        df = pd.read_csv("smoke_detection_iot.csv")

        features = [
            "UTC",
            "Temperature",
            "Humidity",
            "TVOC",
            "eCO2",
            "Raw H2",
            "Raw Ethanol",
            "Pressure",
            "PM1.0",
            "PM2.5",
            "NC0.5",
            "NC1.0",
            "NC2.5",
            "CNT",
        ]
        label = "Fire Alarm"

        df_X = df[features]
        df_y = df.loc[:, label]
        # TODO: END

        folds = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

        pred_y = np.zeros(df_X.shape[0])

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_X, df_y)):
            train_X, train_y = df_X.iloc[train_idx], df_y.iloc[train_idx]
            valid_X, valid_y = df_X.iloc[valid_idx], df_y.iloc[valid_idx]

            clf = XGBClassifier(
                n_jobs=-1,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
            )

            clf.fit(train_X, train_y)

            pred_y[valid_idx] = clf.predict(valid_X)

            del train_X, train_y, valid_X, valid_y
            gc.collect()

        score = f1_score(df_y, pred_y)
        full_accuracy_score = accuracy_score(df_y, pred_y)

        # Log metric to MLflow
        mlflow.log_metric("f1_score", score)
        mlflow.log_metric("accuracy", full_accuracy_score)
        mlflow.xgboost.log_model(clf, artifact_path="models")

        del df_X, df_y, clf
        gc.collect()

    return {"loss": -score, "status": STATUS_OK}


@task
def hypertune_model(df_X, df_y, model):
    """
    Hypertune XGBoost model (the best model from experiment)
    """
    if model == "XGBoost":
        search_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 4, 50, 1)),
            "learning_rate": hp.loguniform("learning_rate", -3, 0),
            "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
            "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        }
        objective_func = objective_xgboost
    else:
        raise Exception("Currently only XGBoost is implemented")

    best_result = fmin(
        fn=objective_func,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
        verbose=False,
    )

    print(best_result)


# Main Function
@flow(task_runner=SequentialTaskRunner())
def main(train_path="smoke_detection_iot.csv"):
    """
    Experiment tracking
    """
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("smoke_detection")

    df_X, df_y = prepare_data(train_path)

    # Exploration training with all models
    train_model_exploration(df_X, df_y, "Random Forest")
    train_model_exploration(df_X, df_y, "LightGBM")
    train_model_exploration(df_X, df_y, "XGBoost")
    train_model_exploration(df_X, df_y, "CatBoost")

    # Hyperparameter tuning
    hypertune_model(df_X, df_y, "XGBoost")


if __name__ == "__main__":
    main(train_path="smoke_detection_iot.csv")
