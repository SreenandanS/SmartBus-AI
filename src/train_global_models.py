# ==============================================================================
# FILE: train_global_models.py
# PURPOSE: Offline trainer – pulls feedback from backend, trains & saves models
# ==============================================================================

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.preprocessing import OneHotEncoder
from .ml_feedback_integration import fetch_training_feedback

DISABILITIES = ["None", "Wheelchair", "Visual Impairment", "Hearing Impairment", "Other"]
MODELS_DIR = os.getenv("MODELS_DIR", "models")

def train_and_save_global_models(since=None):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Pull feedback from backend
    fb = fetch_training_feedback(since)
    seat_recs = fb.get("seattype", [])
    pen_recs  = fb.get("penalty",  [])

    # Encoder (fit on fixed classes so it's stable)
    disability_encoder = OneHotEncoder(handle_unknown='ignore')
    disability_encoder.fit([[d] for d in DISABILITIES])

    # ----- Seat Type Model -----
    seat_type_model = None
    if seat_recs:
        X_seat, y_seat = [], []
        for feat, seat_type in seat_recs:
            onehot = disability_encoder.transform([[feat["disability"]]]).toarray()[0]
            vec = [
                feat["is_priority"],
                feat["is_female"],
                feat["is_in_group"],
                feat["age"],
                feat["norm_row"],
                feat["norm_col"]
            ] + list(onehot)
            X_seat.append(vec)
            y_seat.append(seat_type)

        clf1 = RandomForestClassifier(n_estimators=150, random_state=42)
        clf2 = ExtraTreesClassifier(n_estimators=200, random_state=42)
        seat_type_model = VotingClassifier([("rf", clf1), ("et", clf2)], voting="soft")
        seat_type_model.fit(np.array(X_seat), np.array(y_seat))
        joblib.dump(seat_type_model, os.path.join(MODELS_DIR, "seat_type_model.pkl"))
        print("✅ Saved models/seat_type_model.pkl")

    # ----- Penalty Model -----
    penalty_model = None
    if pen_recs:
        X_pen, y_pen = [], []
        for feat, score in pen_recs:
            onehot = disability_encoder.transform([[feat["disability"]]]).toarray()[0]
            vec = [
                feat["is_priority"],
                feat["is_female"],
                feat["is_in_group"],
                feat["age"],
                feat["norm_row"],
                feat["norm_col"],
                feat["group_distance"]
            ] + list(onehot)
            X_pen.append(vec)
            y_pen.append(score)

        reg1 = LinearRegression()
        reg2 = ExtraTreesRegressor(n_estimators=200, random_state=42)
        penalty_model = VotingRegressor([("lr", reg1), ("et", reg2)])
        penalty_model.fit(np.array(X_pen), np.array(y_pen))
        joblib.dump(penalty_model, os.path.join(MODELS_DIR, "penalty_model.pkl"))
        print("✅ Saved models/penalty_model.pkl")

    # Save encoder
    joblib.dump(disability_encoder, os.path.join(MODELS_DIR, "disability_encoder.pkl"))
    print("✅ Saved models/disability_encoder.pkl")

    return seat_type_model, penalty_model, disability_encoder

if __name__ == "__main__":
    # Optional env var to do incremental training
    SINCE = os.getenv("TRAIN_SINCE")
    train_and_save_global_models(SINCE)
