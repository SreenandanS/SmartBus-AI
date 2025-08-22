# ==============================================================================
# FILE: api.py
# PURPOSE: FastAPI runtime — loads models, personalizes, optimizes,
#          and forwards allocation snapshot to your backend. No feedback handling here.
# ==============================================================================

import os, joblib, uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .integrated_seat_ml_model import Passenger, BusLayout, optimize_seating_with_ml
from .explain_utils import explain_assignment
from .ml_feedback_integration import (
    fetch_passenger_history,
    send_allocation_to_backend,
    train_personalization_model,
)

MODELS_DIR = os.getenv("MODELS_DIR", "models")

# Load global models if available
seat_type_model = penalty_model = disability_encoder = None
if os.path.exists(os.path.join(MODELS_DIR, "seat_type_model.pkl")):
    seat_type_model = joblib.load(os.path.join(MODELS_DIR, "seat_type_model.pkl"))
if os.path.exists(os.path.join(MODELS_DIR, "penalty_model.pkl")):
    penalty_model = joblib.load(os.path.join(MODELS_DIR, "penalty_model.pkl"))
if os.path.exists(os.path.join(MODELS_DIR, "disability_encoder.pkl")):
    disability_encoder = joblib.load(os.path.join(MODELS_DIR, "disability_encoder.pkl"))

app = FastAPI(title="SmartBus-AI", version="1.3.0")

class PassengerReq(BaseModel):
    id: str
    name: Optional[str] = None
    age: int
    gender: str
    disability: str
    groupId: Optional[str] = None
    pickupStopId: int
    dropStopId: int

class VehicleReq(BaseModel):
    rows: int
    columns: int
    vehicleType: Optional[str] = "AC Seater"

class SeatRequest(BaseModel):
    route: Dict[str, Any] = {}
    vehicle: VehicleReq
    passengers: List[PassengerReq]
    tripId: Optional[str] = Field(default=None, description="Optional trip identifier")

@app.get("/health")
def health():
    return {"status": "ok", "models": {
        "seat_type": seat_type_model is not None,
        "penalty": penalty_model is not None,
        "encoder": disability_encoder is not None
    }}

@app.post("/allocate")
def allocate_seats(req: SeatRequest):
    # Generate tripId if not provided
    trip_id = req.tripId or uuid.uuid4().hex

    # Build bus & passengers
    bus = BusLayout(rows=req.vehicle.rows, cols=req.vehicle.columns, accessibility_rows=3, aisle_after=2)
    passengers = [
        Passenger(
            id=p.id, age=p.age, gender=p.gender, disability=p.disability,
            group_id=p.groupId, source_stop=p.pickupStopId, dest_stop=p.dropStopId
        )
        for p in req.passengers
    ]

    # Personalization models per passenger (from backend histories)
    personal_models = {}
    for p in passengers:
        history = fetch_passenger_history(p.id)
        if history:
            m = train_personalization_model(history)
            if m:
                personal_models[p.id] = m

    # Run optimization
    assignments = optimize_seating_with_ml(
        passengers, bus,
        seat_type_model=seat_type_model,
        penalty_model=penalty_model,
        personal_models=personal_models,
        disability_encoder=disability_encoder
    )

    if not assignments:
        raise HTTPException(status_code=409, detail="No feasible seating assignment found.")

    # Build response & snapshot
    results = []
    for p in passengers:
        details = assignments.get(p.id)
        if not details:
            continue
        seat = bus.seat_map[details["seat_id"]]
        u = details["universal_features"]
        explanation = explain_assignment(
            passenger=p,
            seat=seat,
            bus_layout=bus,
            predicted_type=u.get("seat_type"),
            is_personalized=p.id in personal_models
        )
        results.append({
            "tripId": trip_id,
            "passengerId": p.id,
            "groupId": p.group_id,
            "seatId": details["seat_id"],
            "groupDistance": u.get("group_distance"),
            "seatType": u.get("seat_type"),
            "normRow": u.get("norm_row"),
            "normCol": u.get("norm_col"),
            "explanation": explanation
        })

    # Forward allocation snapshot to backend
    try:
        send_allocation_to_backend(
            trip_id=trip_id,
            vehicle={"rows": req.vehicle.rows, "columns": req.vehicle.columns, "vehicleType": req.vehicle.vehicleType},
            assignments_payload=results
        )
    except Exception as e:
        print(f"[WARN] Allocation forwarding failed: {e}")

    return {"assignments": results, "tripId": trip_id}

