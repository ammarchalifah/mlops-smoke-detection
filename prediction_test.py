import requests

smoke = {
    "UTC": 1654733331,
    "Temperature": 20.0,
    "Humidity": 57.0,
    "TVOC": 0,
    "eCO2": 400,
    "Raw H2": 12306,
    "Raw Ethanol": 18520,
    "Pressure": 939.0,
    "PM1.0": 0.0,
    "PM2.5": 0.0,
    "NC0.5": 0.0,
    "NC1.0": 0.0,
    "NC2.5": 0.0,
    "CNT": 0,
}

print(smoke.values())

url = "http://localhost:9696/predict"
response = requests.post(url, json=smoke)
print(response.json())
