from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.utils.schema import SeasonEnum

app = FastAPI()

DATABASE_URL = "postgresql+psycopg2://postgres:root@localhost:5432/kisanmitra"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()



Base.metadata.create_all(bind=engine)

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

class FarmerSchema(BaseModel):
    year: int
    area: float
    production: float = 0.0
    yield_per_ha: float = 0.0
    irrigation: float = 50.0
    crop: str
    season: SeasonEnum

@app.post("/add_farmer")
def add_farmer(farmer: FarmerSchema):
    db = SessionLocal()
    try:
        new_farmer = Farmer(
            year=farmer.year,
            area=farmer.area,
            production=farmer.production,
            yield_per_ha=farmer.yield_per_ha,
            irrigation=farmer.irrigation,
            crop=farmer.crop,            # corrected here
            season=farmer.season
        )
        db.add(new_farmer)
        db.commit()
        db.refresh(new_farmer)
        return {"message": "Farmer added successfully", "farmer_id": new_farmer.id}
    finally:
        db.close()
