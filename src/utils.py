# ==============================================================================
# FILE: utils.py
# PURPOSE: Centralized utility functions
# ==============================================================================

def get_seat_type(seat, bus_layout):
    """
    Classify seat type (front, window, aisle) based on seat position & layout.
    """
    if seat.row < bus_layout.accessibility_rows:
        return 'front'
    if seat.col == 0 or seat.col == bus_layout.cols - 1:
        return 'window'
    return 'aisle'
