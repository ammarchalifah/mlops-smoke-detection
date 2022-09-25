FROM prefecthq/prefect:2.3.1-python3.10

# Install Python packages
RUN apt-get update
RUN apt-get install libgomp1
RUN pip install mlflow boto3 pymysql scikit-learn jupyter pandas scipy xgboost catboost numpy hyperopt prefect lightgbm

# FROM python:3.7-slim-buster

# # Set Directories
# RUN mkdir /jupyter-notebooks
# RUN mkdir /prefect
# WORKDIR /prefect

# # Install Python packages
# RUN apt-get update
# RUN apt-get install libgomp1
# RUN pip install mlflow boto3 pymysql scikit-learn jupyter pandas scipy xgboost catboost numpy hyperopt prefect lightgbm

# # Copy Entrypoint script
# COPY ./entrypoint.sh /entrypoint.sh
