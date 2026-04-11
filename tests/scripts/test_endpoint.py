"""
Send rows from future_unseen_examples.csv to the prediction endpoint one by one,
with a 10-second pause between each request.
"""

import csv
import time
import requests

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/predict"
CSV_PATH = "data/future_unseen_examples.csv"
DELAY_SECONDS = 0


def send_row(row: dict, index: int) -> None:
    payload = {
        "bedrooms": int(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "sqft_living": float(row["sqft_living"]),
        "sqft_lot": float(row["sqft_lot"]),
        "floors": float(row["floors"]),
        "waterfront": int(row["waterfront"]),
        "view": int(row["view"]),
        "condition": int(row["condition"]),
        "grade": int(row["grade"]),
        "sqft_above": float(row["sqft_above"]),
        "sqft_basement": float(row["sqft_basement"]),
        "yr_built": int(row["yr_built"]),
        "yr_renovated": int(row["yr_renovated"]),
        "zipcode": str(row["zipcode"]).zfill(5),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "sqft_living15": float(row["sqft_living15"]),
        "sqft_lot15": float(row["sqft_lot15"]),
    }

    try:
        response = requests.post(ENDPOINT, json=payload, timeout=10)
        print(f"[{index}] Status: {response.status_code} | Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[{index}] Request failed: {e}")


def main():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            send_row(row, i)
            if i < 100:  # no sleep after the last row
                time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
