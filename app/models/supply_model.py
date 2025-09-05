import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "persisted_models", "supply.pkl")

class SupplyPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None

    def prepare_supply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.rename(columns={
            'year': 'year',
            'area_million_hectares': 'area',
            'production_million_tonnes': 'production',
            'yield_tonnes_per_hectare': 'yield_per_ha',
            'temperature_celsius': 'temp',
            'fertilizer_kg_per_hectare': 'fertilizer',
            'rainfal_mm': 'rainfall',
            'irrigation_pct': 'irrigation',
            'crop': 'crop'
        }).copy()
        data['rainfall_adequacy'] = np.clip(data['rainfall'] / 1000, 0.5, 1.5)
        data['temperature_stress'] = (data['temp'] > 40).astype(int)
        data['irrigation_factor'] = data['irrigation'] / 100
        data['fertilizer_intensity'] = data['fertilizer'] / 150
        data['tech_adoption'] = (data['year'] - 2009) * 0.03
        data['crop_rice'] = (data['crop'] == 'Rice').astype(int)
        data['crop_wheat'] = (data['crop'] == 'Wheat').astype(int)
        data['crop_pulse'] = data['crop'].isin(['Gram','Tur','Urad','Moong','Masoor']).astype(int)
        return data

    def train_and_persist(self, df_prepared: pd.DataFrame, persist=True):
        print("🌾 Training Supply Prediction Model...")
        self.feature_cols = [
            'area', 'rainfall_adequacy', 'temperature_stress',
            'irrigation_factor', 'fertilizer_intensity', 'tech_adoption',
            'crop_rice', 'crop_wheat', 'crop_pulse'
        ]
        X = df_prepared[self.feature_cols]
        y = df_prepared['production']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_split=3, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        print(f"Supply Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
        print(f"Supply Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
        print(f"Supply Test R²: {r2_score(y_test, test_pred):.3f}")

        self.model = model

        if persist:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols
            }, MODEL_PATH)
            print(f"Saved supply model to {MODEL_PATH}")

        return self.model

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Supply model not found. Train first or provide model.")
        obj = joblib.load(MODEL_PATH)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        self.feature_cols = obj["feature_cols"]
        return self.model

    def predict_row(self, row: dict):
        if self.model is None:
            raise ValueError("Supply model not loaded.")
        X = pd.DataFrame([row], columns=self.feature_cols)
        X_scaled = self.scaler.transform(X) # type: ignore
        pred = self.model.predict(X_scaled)[0]
        return float(max(0, pred))
