# SmartBus-AI

SmartBus-AI allocates bus seats using a **MILP optimizer** (hard constraints + strong group cohesion) enhanced by **ML signals**:

- **Global models** (trained offline from backend’s feedback data)
- **Personalized runtime models** (tiny per-passenger regressors using their own history)

This repo exposes a FastAPI service (`/allocate`) and includes an offline trainer to build global models. All data persistence is handled by **backend**; this service is stateless.

---

## Architecture

### Optimizer (MILP)
- One seat per passenger.
- No seat reuse when passenger trip intervals overlap.
- Safety zones:  
  - Male passengers cannot sit in female-only rows.  
  - Non-priority (not disabled & under 60) cannot sit in accessible rows.  
  - Wheelchair users must be in accessible rows.
- Objective: strong **group cohesion** (horizontal > vertical), with optional ML bonuses:
  - Global **seat type** preference (front/window/aisle).
  - Global **penalty** for group separation distance.
  - **Personalized** bonus if the passenger historically rates similar seat locations higher.

### ML
- **Offline global models** (trained with `src/train_global_models.py`) — saved into `models/`.
- **Runtime personalization** per passenger (if history available from backend).

---
