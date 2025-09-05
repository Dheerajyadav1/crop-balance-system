from fastapi import APIRouter, HTTPException
from app.utils.schema import Request, SeasonEnum
from app.models.demand_model import DemandPredictionModel
from app.models.supply_model import SupplyPredictionModel
from app.models.balance_system import SupplyDemandBalanceSystem
from sqlalchemy import create_engine, Column, Integer, String, Float, Enum, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

router = APIRouter()

# Database config
# DATABASE_URL = "postgresql+psycopg2://postgres:root@localhost:5432/kisanmitra"
# engine = create_engine(DATABASE_URL, echo=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database config
DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_nyYXfN5qaMG0@ep-bold-morning-adlow3e3-pooler.c-2.us-east-1.aws.neon.tech/kisanmitra?sslmode=require"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Base = declarative_base()


# Farmer table
class Farmer(Base):
    __tablename__ = "farmers"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, index=True)
    area = Column(Float)
    production = Column(Float)
    yield_per_ha = Column(Float)
    irrigation = Column(Float)
    crop = Column(String, index=True)
    season = Column(Enum(SeasonEnum), index=True)


# Create tables if not exist
Base.metadata.create_all(bind=engine)


# Load models once
dpm = DemandPredictionModel()
spm = SupplyPredictionModel()
sdbs = SupplyDemandBalanceSystem(demand_model=dpm, supply_model=spm)

try:
    dpm.load()
except Exception:
    pass

try:
    spm.load()
except Exception:
    pass


# POST → predict demand & supply, save to DB, return balance
@router.post("/predict")
def predict(payload: Request):
    # --- Demand ---
    if dpm.feature_cols is None:
        raise HTTPException(status_code=500, detail="Demand model not loaded. Train first.")

    features = {}
    for col in dpm.feature_cols:
        if col == "crop_encoded":
            try:
                features[col] = int(dpm.crop_encoder.transform([payload.crop])[0])  # type: ignore
            except Exception:
                features[col] = 0
        else:
            val = getattr(payload, col, None)
            if val is None:
                defaults = {
                    "temp": 25.0,
                    "rainfall": 800.0,
                    "irrigation": payload.irrigation,
                    "year": payload.year,
                    "season": payload.season,
                }
                val = defaults.get(col, 0.0)
            features[col] = val

    demand_pred = dpm.predict_row(features)

    # --- Supply ---
    if spm.feature_cols is None:
        raise HTTPException(status_code=500, detail="Supply model not loaded. Train first.")

    feat = {
        "area": payload.area,
        "rainfall_adequacy": 800.0 / 1000,
        "temperature_stress": 1,
        "irrigation_factor": payload.irrigation / 100,
        "fertilizer_intensity": payload.fertilizer / 150,
        "tech_adoption": (payload.year - 2009) * 0.03,
        "crop_rice": 1 if payload.crop == "Rice" else 0,
        "crop_wheat": 1 if payload.crop == "Wheat" else 0,
        "crop_pulse": 1 if payload.crop in ["Gram", "Tur", "Urad", "Moong", "Masoor"] else 0,
    }
    # supply_pred = spm.predict_row(feat)

    # --- Save farmer record ---
    db = SessionLocal()
    supply_pred = db.execute(
        text("SELECT SUM(production) FROM farmers WHERE crop = :crop AND season = :season"),
        {"crop": payload.crop, "season": payload.season.value}
    ).scalar() or 0.0
    try:
        new_farmer = Farmer(
            year=payload.year,
            area=payload.area,
            production=payload.production,
            yield_per_ha=payload.yield_per_ha,
            irrigation=payload.irrigation,
            crop=payload.crop,
            season=payload.season,
        )
        db.add(new_farmer)
        db.commit()
        db.refresh(new_farmer)
        supply_pred = db.execute(
            text("SELECT SUM(production) FROM farmers WHERE crop = :crop AND season = :season"),
            {"crop": payload.crop, "season": payload.season.value}
        ).scalar() or 0.0

    finally:
        db.close()

    # --- Return balance result ---
    return {
        "status": sdbs.calculate_balance(payload.crop, payload.area, demand_pred, supply_pred),
        "total_production": supply_pred
    }
