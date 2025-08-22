
# ==============================================================================
# FILE: explain_utils.py
# PURPOSE: Transparent reasoning for a seat assignment
# ==============================================================================

from .utils import get_seat_type

def explain_assignment(passenger, seat, bus_layout, predicted_type=None, is_personalized=False):
    explanation = [f"Passenger {passenger.id} (Age: {passenger.age}) -> Seat {seat.id}"]

    if is_personalized:
        explanation.append("• Personalized choice based on past history.")
    if passenger.group_id:
        explanation.append("• Group cohesion prioritized.")

    actual_type = get_seat_type(seat, bus_layout)
    if predicted_type and predicted_type == actual_type:
        explanation.append(f"• Global model matched seat type: {predicted_type}.")

    if seat.zone == 'female_only' and str(passenger.gender).lower() == 'female':
        explanation.append("• Female-only zone respected for safety.")

    if seat.zone == 'disability':
        if passenger.disability.lower() == 'wheelchair':
            explanation.append("• Accessible seat required for Wheelchair (mandatory).")
        elif passenger.disability.lower() != 'none':
            explanation.append(f"• Accessible seat for {passenger.disability}.")
        elif passenger.age >= 60:
            explanation.append("• Senior priority in accessible zone.")

    if len(explanation) == 1:
        explanation.append("• Standard assignment based on availability.")

    return "\n".join(explanation)
