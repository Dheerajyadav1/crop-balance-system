from pydantic import BaseModel
from typing import Optional, List
# from sqlalchemy import create_engine, Column, Integer, String, Float, Enum

import enum

class SeasonEnum(enum.Enum):
    kharif = "kharif"
    rabi = "rabi"
    zaid = "zaid"

class Request(BaseModel):
    year: int
    area: float
    production: float = 0.0
    yield_per_ha: float = 0.0
    # temp: float = 25.0
    fertilizer: float = 100.0
    # rainfall: float = 800.0
    irrigation: float = 50.0
    crop: str
    season: SeasonEnum
