import requests

payload = {
    "method": "cqr",
    "median_income": 3.2,
    "house_age": 20,
    "average_rooms": 5.1,
    "average_bedrooms": 1.1,
    "population": 1200,
    "average_occupancy": 2.8,
    "latitude": 34.2,
    "longitude": -118.4
}

response = requests.post(
    "http://127.0.0.1:8000/predict-housing-price",
    json=payload
)

print(response.status_code)
print(response.json())
