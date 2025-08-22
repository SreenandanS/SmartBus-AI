# ==============================================================================
# FILE: ml_feedback_integration.py
# PURPOSE: Backend integration hooks + personalization trainer
# ==============================================================================

import os
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

# Backend base URL (override with env var)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")

# ---------------- Backend Hooks ----------------

def fetch_passenger_history(passenger_id):
    """
    GET {BACKEND_BASE_URL}/histories/{passenger_id}
    Expected response: list of dicts with keys: norm_row, norm_col, feedback_score
    """
    url = f"{BACKEND_BASE_URL}/histories/{passenger_id}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[WARN] fetch_passenger_history failed for {passenger_id}: {e}")
        return []

def send_allocation_to_backend(trip_id, vehicle, assignments_payload):
    """
    POST {BACKEND_BASE_URL}/allocations
    Payload example:
    {
      "tripId": "...",
      "vehicle": {"rows":12,"columns":5,"vehicleType":"AC Seater"},
      "assignments":[
        {"passengerId":"uuid-1","seatId":"1A","seatType":"front","normRow":0.0,"normCol":0.0,"groupDistance":0,"groupId":null,"explanation":"..."}
      ]
    }
    """
    url = f"{BACKEND_BASE_URL}/allocations"
    payload = {
        "tripId": trip_id,
        "vehicle": vehicle,
        "assignments": assignments_payload
    }
    try:
        resp = requests.post(url, json=payload, timeout=8)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] send_allocation_to_backend failed: {e}")

def fetch_training_feedback(since=None):
    """
    GET {BACKEND_BASE_URL}/feedback?since=YYYY-MM-DD
    Expected response:
      { "seattype": [(features, seat_type_str), ...],
        "penalty":  [(features, feedback_score_float), ...] }
    """
    url = f"{BACKEND_BASE_URL}/feedback"
    params = {"since": since} if since else {}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"seattype": [], "penalty": []}
    except Exception as e:
        print(f"[WARN] fetch_training_feedback failed: {e}")
        return {"seattype": [], "penalty": []}

# ---------------- Personalization ----------------

def train_personalization_model(passenger_history):
    """
    Train per-passenger tiny regressor using normalized seat (row/col) -> feedback_score.
    """
    if not passenger_history or len(passenger_history) < 3:
        return None
    X = np.array([[rec['norm_row'], rec['norm_col']] for rec in passenger_history])
    y = np.array([rec['feedback_score'] for rec in passenger_history])
    model = LinearRegression()
    model.fit(X, y)
    return model

