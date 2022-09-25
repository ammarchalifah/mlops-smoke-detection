#!/bin/bash

cd /prefect

prefect deployment build flows/model_training.py:main -n model_training -q ml_ops
prefect deployment apply main-deployment.yaml
prefect orion start &
prefect agent start -q ml_ops
