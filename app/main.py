from fastapi import FastAPI
from app.routers.balance import router as balance_router
from train_models import main as train_models_main

train_models_main()

app = FastAPI(title="🌾 Crop Supply-Demand Balance API")

# Mount all balance routes under "/predict"
app.include_router(balance_router, prefix="", tags=["Crop Balance"])

@app.get("/")
def root():
    return {"message": "🌾 Crop Supply-Demand Balance API running!"}
