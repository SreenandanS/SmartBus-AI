# ==============================================================================
# FILE: integrated_seat_ml_model.py
# PURPOSE: MILP + ML seat optimizer (group cohesion + zones + ML bonuses)
# ==============================================================================

import pulp
import itertools
import collections
import numpy as np
from .utils import get_seat_type

# -------------------- Data Structures --------------------

class Passenger:
    def __init__(self, id, age=30, group_id=None, gender="Male", disability="None",
                 source_stop=0, dest_stop=10):
        self.id = id
        self.age = age
        self.group_id = group_id
        self.gender = gender
        self.disability = disability
        self.is_disabled = self.disability.lower() != 'none'
        self.requires_disability_zone = self.disability.lower() == 'wheelchair'
        self.source_stop = source_stop
        self.dest_stop = dest_stop

class Seat:
    def __init__(self, seat_id, row, col, is_accessible=False, zone="general"):
        self.id = seat_id
        self.row = row
        self.col = col
        self.is_accessible = is_accessible
        self.zone = zone

class BusLayout:
    def __init__(self, rows, cols, accessibility_rows=2, gender_zones=None, aisle_after=1):
        if gender_zones is None:
            gender_zones = {"female": [2, 3]}
        self.rows = rows
        self.cols = cols
        self.aisle_after = aisle_after
        self.accessibility_rows = accessibility_rows
        self.gender_zones = gender_zones
        self.seats = self._generate_seats()
        self.seat_map = {s.id: s for s in self.seats}
        self.adjacent_pairs = self._compute_adjacents()

    def _generate_seats(self):
        seats = []
        for r in range(self.rows):
            for c in range(self.cols):
                sid = f"{r+1}{chr(65+c)}"
                accessible = r < self.accessibility_rows
                zone = "disability" if accessible else ("female_only" if r in self.gender_zones.get("female", []) else "general")
                seats.append(Seat(sid, r, c, accessible, zone))
        return seats

    def _compute_adjacents(self):
        """
        Adjacent pairs: horizontal (not across the aisle) and vertical.
        """
        adj = set()
        seats_by_pos = {(s.row, s.col): s.id for s in self.seats}
        for s in self.seats:
            # horizontal (avoid across aisle)
            if (s.row, s.col + 1) in seats_by_pos and s.col + 1 != self.aisle_after:
                adj.add(tuple(sorted((s.id, seats_by_pos[(s.row, s.col + 1)]))))
            # vertical
            if (s.row + 1, s.col) in seats_by_pos:
                adj.add(tuple(sorted((s.id, seats_by_pos[(s.row + 1, s.col)]))))
        return list(adj)

# -------------------- MILP with ML Adjustments --------------------

def optimize_seating_with_ml(passengers, bus, seat_type_model=None, penalty_model=None, personal_models=None, disability_encoder=None):
    """
    Run MILP and return dict keyed by passenger id:
      { passenger_id: { "seat_id": "3A", "universal_features": {...} } }
    ML models are optional (seat_type_model, penalty_model, disability_encoder). personal_models is a dict.
    """
    personal_models = personal_models or {}
    model = pulp.LpProblem("SeatAllocation_ML", pulp.LpMaximize)
    assign = pulp.LpVariable.dicts("x", ((p.id, s.id) for p in passengers for s in bus.seats), cat="Binary")

    # --- Hard Constraints ---
    for p in passengers:
        model += pulp.lpSum(assign[(p.id, s.id)] for s in bus.seats) == 1, f"oneSeat_{p.id}"

    for s in bus.seats:
        for p1, p2 in itertools.combinations(passengers, 2):
            # overlapping intervals cannot share seat
            if not (p1.dest_stop <= p2.source_stop or p2.dest_stop <= p1.source_stop):
                model += assign[(p1.id, s.id)] + assign[(p2.id, s.id)] <= 1, f"noOverlap_{s.id}_{p1.id}_{p2.id}"

    for p in passengers:
        for s in bus.seats:
            if str(p.gender).lower() == "male" and s.zone == "female_only":
                model += assign[(p.id, s.id)] == 0
            if not (p.is_disabled or p.age >= 60) and s.zone == "disability":
                model += assign[(p.id, s.id)] == 0
            if p.requires_disability_zone and s.zone != "disability":
                model += assign[(p.id, s.id)] == 0

    # --- Objective ---
    obj = []
    group_dict = collections.defaultdict(list)
    for p in passengers:
        if p.group_id:
            group_dict[p.group_id].append(p)

    # Strong group adjacency bonuses
    for gid, members in group_dict.items():
        for p1, p2 in itertools.combinations(members, 2):
            for s1_id, s2_id in bus.adjacent_pairs:
                adj_var = pulp.LpVariable(f"adj_{p1.id}_{p2.id}_{s1_id}_{s2_id}", cat="Binary")
                model += adj_var >= assign[(p1.id, s1_id)] + assign[(p2.id, s2_id)] - 1
                s1_obj, s2_obj = bus.seat_map[s1_id], bus.seat_map[s2_id]
                bonus = 3000 if s1_obj.row == s2_obj.row else 1500
                obj.append(bonus * adj_var)

    # ML-informed group separation penalty (soft)
    if penalty_model and disability_encoder:
        for gid, members in group_dict.items():
            for p1, p2 in itertools.combinations(members, 2):
                for s1 in bus.seats:
                    for s2 in bus.seats:
                        dist = abs(s1.row - s2.row) + abs(s1.col - s2.col)
                        if dist > 1:
                            far_var = pulp.LpVariable(f"far_{p1.id}_{s1.id}_{p2.id}_{s2.id}", cat="Binary")
                            model += far_var >= assign[(p1.id, s1.id)] + assign[(p2.id, s2.id)] - 1
                            norm_s1_row = s1.row / (bus.rows - 1) if bus.rows > 1 else 0.0
                            norm_s1_col = s1.col / (bus.cols - 1) if bus.cols > 1 else 0.0
                            try:
                                onehot = disability_encoder.transform([[p1.disability]]).toarray()[0]
                            except Exception:
                                onehot = [0] * (len(disability_encoder.categories_[0]) if hasattr(disability_encoder, 'categories_') else 1)
                            features = [int(p1.is_disabled or p1.age >= 60),
                                        int(str(p1.gender).lower() == 'female'),
                                        1,  # in group
                                        p1.age, norm_s1_row, norm_s1_col, dist] + list(onehot)
                            try:
                                predicted = float(penalty_model.predict([features])[0])
                                penalty = max(0.0, 5.0 - predicted)
                            except Exception:
                                penalty = 5.0
                            obj.append(-penalty * dist * far_var)

    # Global seat-type preference bonus
    if seat_type_model and disability_encoder:
        for p in passengers:
            for s in bus.seats:
                norm_s_row = s.row / (bus.rows - 1) if bus.rows > 1 else 0.0
                norm_s_col = s.col / (bus.cols - 1) if bus.cols > 1 else 0.0
                try:
                    onehot = disability_encoder.transform([[p.disability]]).toarray()[0]
                except Exception:
                    onehot = [0] * (len(disability_encoder.categories_[0]) if hasattr(disability_encoder, 'categories_') else 1)
                vec = [int(p.is_disabled or p.age >= 60),
                       int(str(p.gender).lower() == 'female'),
                       int(p.group_id is not None),
                       p.age, norm_s_row, norm_s_col] + list(onehot)
                try:
                    probs = seat_type_model.predict_proba([vec])[0]
                    preferred = seat_type_model.classes_[int(np.argmax(probs))]
                except Exception:
                    try:
                        preferred = seat_type_model.predict([vec])[0]
                    except Exception:
                        preferred = None
                if preferred and get_seat_type(s, bus) == preferred:
                    obj.append(assign[(p.id, s.id)] * 50)

    # Personalization bonus (per passenger tiny model)
    for p in passengers:
        personal_model = personal_models.get(p.id)
        if not personal_model:
            continue
        for s in bus.seats:
            norm_s_row = s.row / (bus.rows - 1) if bus.rows > 1 else 0.0
            norm_s_col = s.col / (bus.cols - 1) if bus.cols > 1 else 0.0
            try:
                personal_score = float(personal_model.predict([[norm_s_row, norm_s_col]])[0])
                bonus = max(0.0, personal_score - 3.0) * 100.0
                obj.append(assign[(p.id, s.id)] * bonus)
            except Exception:
                pass

    model += pulp.lpSum(obj)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    if model.status != pulp.LpStatusOptimal:
        return {}

    # Collect assignment mapping
    chosen = {}
    for (p_id, s_id), var in assign.items():
        if pulp.value(var) == 1:
            chosen[p_id] = s_id

    # Compute group distances & universal features for DB
    group_distances = {}
    for gid, members in group_dict.items():
        seats = [bus.seat_map[chosen[p.id]] for p in members]
        maxd = 0
        for i in range(len(seats)):
            for j in range(i+1, len(seats)):
                d = abs(seats[i].row - seats[j].row) + abs(seats[i].col - seats[j].col)
                if d > maxd:
                    maxd = d
        group_distances[gid] = maxd

    assignment_details = {}
    for p in passengers:
        seat = bus.seat_map[chosen[p.id]]
        assignment_details[p.id] = {
            "seat_id": seat.id,
            "universal_features": {
                "seat_type": get_seat_type(seat, bus),
                "norm_row": seat.row / (bus.rows - 1) if bus.rows > 1 else 0.0,
                "norm_col": seat.col / (bus.cols - 1) if bus.cols > 1 else 0.0,
                "group_distance": group_distances.get(p.group_id)
            }
        }

    return assignment_details

