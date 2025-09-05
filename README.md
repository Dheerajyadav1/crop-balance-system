# Crop Balance System (FastAPI)

## Overview
API with demand, supply, and supply-demand balance endpoints. Models are trained from `app/data/final_data.csv` and persisted to `app/persisted_models/`.

## Setup (local)
1. Create virtualenv and install:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Place `final_data.csv` under `app/data/`.
3. Train models:python train_models.py
4. Run API: uvicorn app.main:app --reload
5. Open docs: http://127.0.0.1:8000/docs

## Endpoints
- `POST /demand/predict` -> predict demand (payload: DemandRequest)
- `POST /demand/train` -> train demand model (payload: {"data_path": ...})
- `POST /supply/predict` -> predict supply (payload: SupplyRequest)
- `POST /supply/train` -> train supply model
- `POST /balance/calculate` -> calculate balance for scenarios (payload: BalanceRequest)

## Deploy (Render)
- Push repo to GitHub and connect to Render.
- Use `render.yaml` above or set start command to:
`uvicorn app.main:app --host 0.0.0.0 --port 10000`
