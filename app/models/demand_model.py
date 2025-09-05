import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "persisted_models", "demand.pkl")

class DemandPredictionModel:
    def __init__(self):
        self.demand_model = None
        self.crop_encoder = None
        self.scaler = None
        self.feature_cols = None

    def prepare_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.rename(columns={
            'year': 'year',
            'area_million_hectares': 'area',
            'production_million_tonnes': 'production',
            'yield_tonnes_per_hectare': 'yield_per_ha',
            'temperature_celsius': 'temp',
            'fertilizer_kg_per_hectare': 'fertilizer',
            'rainfal_mm': 'rainfall',
            'irrigation_pct': 'irrigation',
            'optimal_requirements': 'optimal_production',
            'crop': 'crop'
        }).copy()
        data = data.dropna(subset=['optimal_production'])
        data['population_growth'] = (data['year'] - 2009) * 0.012
        data['urbanization_rate'] = 35 + (data['year'] - 2009) * 0.5
        data['gdp_per_capita'] = 1500 + (data['year'] - 2009) * 120
        data['inflation_rate'] = 5.0
        data['export_demand'] = 1.1
        data['season_kharif'] = data['crop'].isin(['Rice','Maize','Cotton','Sugarcane']).astype(int)
        data['season_rabi'] = data['crop'].isin(['Wheat','Gram','Rapeseed & Mustard']).astype(int)
        return data

    def train_and_persist(self, df_prepared: pd.DataFrame, persist=True):
        le = LabelEncoder()
        df_prepared['crop_encoded'] = le.fit_transform(df_prepared['crop'])
        self.crop_encoder = le

        self.feature_cols = [
            'year', 'temp', 'rainfall', 'irrigation',
            'population_growth', 'urbanization_rate', 'gdp_per_capita',
            'inflation_rate', 'export_demand',
            'season_kharif', 'season_rabi',
            'crop_encoded'
        ]
        X = df_prepared[self.feature_cols]
        y = df_prepared['optimal_production']

        imp = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imp.fit_transform(X), columns=self.feature_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        print(f"Demand Model testing R²: {r2_score(y_test, y_test_pred):.3f}, MAE: {mean_absolute_error(y_test, y_test_pred):.3f}")

        self.demand_model = model

        if persist:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump({
                "model": self.demand_model,
                "scaler": self.scaler,
                "crop_encoder": self.crop_encoder,
                "feature_cols": self.feature_cols
            }, MODEL_PATH)
            print(f"Saved demand model to {MODEL_PATH}")

        return self.demand_model

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Demand model not found. Train first or provide model.")
        obj = joblib.load(MODEL_PATH)
        self.demand_model = obj["model"]
        self.scaler = obj["scaler"]
        self.crop_encoder = obj["crop_encoder"]
        self.feature_cols = obj["feature_cols"]
        return self.demand_model

    def predict_row(self, row: dict):
        if self.demand_model is None:
            raise ValueError("Demand model not loaded.")
        import pandas as pd
        X = pd.DataFrame([row], columns=self.feature_cols)
        # impute if necessary
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imp.fit_transform(X), columns=self.feature_cols)
        X_scaled = self.scaler.transform(X) # type: ignore
        pred = self.demand_model.predict(X_scaled)[0]
        return float(max(0, pred))
