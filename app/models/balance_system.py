from typing import List, Dict
import numpy as np

class SupplyDemandBalanceSystem:
    def __init__(self, demand_model, supply_model):
        """
        demand_model: instance of DemandPredictionModel (loaded)
        supply_model: instance of SupplyPredictionModel (loaded)
        """
        self.demand_model = demand_model
        self.supply_model = supply_model

    def calculate_balance(self, crop, area, demand_forecast, supply_forecast):
        results = []
        balance_ratio = supply_forecast / demand_forecast if demand_forecast > 0 else 0
        surplus_deficit = supply_forecast - demand_forecast
        action_data = self._calculate_incentive(crop, balance_ratio, surplus_deficit, demand_forecast)
        results.append({
            'crop': crop,
            'demand_forecast_MT': round(demand_forecast, 2),
            'supply_forecast_MT': round(supply_forecast, 2),
            'balance_ratio': round(balance_ratio, 3),
            'surplus_deficit_MT': round(surplus_deficit, 2),
            **action_data,
            'area_hectares': area
        })
        return results

    def _calculate_incentive(self, crop, balance_ratio, surplus_deficit, demand_forecast):
        base_incentives = {
            'Rice': 15000, 'Wheat': 12000, 'Maize': 10000,
            'Gram': 18000, 'Tur': 20000,
            'Groundnut': 14000, 'Soybean': 16000,
            'Cotton': 22000, 'Sugarcane': 25000
        }
        base_incentive = base_incentives.get(crop, 12000)
        surplus_deficit_val = float(surplus_deficit)
        if balance_ratio < 0.85:
            severity = (0.85 - balance_ratio) / 0.85
            incentive_multiplier = 1 + (severity * 1.5)
            incentive_amount = base_incentive * incentive_multiplier
            action = "increase_cultivation"
            urgency = "high" if balance_ratio < 0.7 else "medium"
            message = f"🚨 UNDERSUPPLY: Increase {crop} cultivation. {abs(surplus_deficit_val):.1f} MT shortage expected!"
        elif balance_ratio > 1.20:
            severity = (balance_ratio - 1.20) / 0.50
            penalty_multiplier = min(severity * 0.8, 0.6)
            incentive_amount = base_incentive * (1 - penalty_multiplier)
            action = "diversify_or_reduce"
            urgency = "medium" if balance_ratio < 1.5 else "high"
            message = f"⚠️  OVERSUPPLY: Consider alternative crops. {surplus_deficit_val:.1f} MT surplus expected!"
        elif 0.95 <= balance_ratio <= 1.05:
            incentive_amount = base_incentive * 1.1
            action = "maintain_current"
            urgency = "low"
            message = f"✅ BALANCED: Perfect supply-demand match. Continue current cultivation."
        else:
            incentive_amount = base_incentive
            action = "maintain_current"
            urgency = "low"
            message = f"✅ GOOD: Supply-demand within acceptable range."
        return {
            'action': action,
            'urgency': urgency,
            'incentive_per_hectare_inr': round(incentive_amount, 0),
            'message': message
        }

    def generate_recommendations(self, balance_results):
        recommendations = {
            'increase_cultivation': [],
            'reduce_cultivation': [],
            'maintain_current': [],
            'alternative_crops': []
        }
        for result in balance_results:
            crop_info = {
                'crop': result['crop'],
                'deficit_surplus': result['surplus_deficit_MT'],
                'incentive': result['incentive_per_hectare_inr'],
                'urgency': result['urgency']
            }
            if result['action'] == 'increase_cultivation':
                recommendations['increase_cultivation'].append(crop_info)
            elif result['action'] == 'diversify_or_reduce':
                recommendations['reduce_cultivation'].append(crop_info)
                # alternatives
                alternatives = self._suggest_alternative_crops(result['crop'])
                recommendations['alternative_crops'].extend(alternatives)
            else:
                recommendations['maintain_current'].append(crop_info)
        return recommendations

    def _suggest_alternative_crops(self, oversupplied_crop):
        alternatives = {
            'Rice': ['Maize', 'Gram', 'Groundnut'],
            'Wheat': ['Gram', 'Rapeseed', 'Tur'],
            'Cotton': ['Groundnut', 'Soybean', 'Gram'],
            'Sugarcane': ['Rice', 'Maize', 'Soybean'],
            'Maize': ['Soybean', 'Gram', 'Sunflower']
        }
        return alternatives.get(oversupplied_crop, ['Gram', 'Soybean'])
