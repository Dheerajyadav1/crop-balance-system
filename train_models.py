import sys
sys.stdout.reconfigure(encoding='utf-8') # type: ignore

"""
Train both demand and supply models locally and persist them.
Usage: python train_models.py
"""
import os
import pandas as pd
from app.models.demand_model import DemandPredictionModel
from app.models.supply_model import SupplyPredictionModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "app", "data", "final_data.csv")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Place final_data.csv there.")
    df = pd.read_csv(DATA_PATH)

    # Demand
    dpm = DemandPredictionModel()
    df_prepared = dpm.prepare_demand_features(df)
    dpm.train_and_persist(df_prepared, persist=True)

    # Supply
    spm = SupplyPredictionModel()
    df_supply = spm.prepare_supply_features(df)
    spm.train_and_persist(df_supply, persist=True)

if __name__ == "__main__":
    main()
