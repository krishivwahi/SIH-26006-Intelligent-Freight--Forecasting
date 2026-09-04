import json
import random
from datetime import date, timedelta
import os

def generate_dummy_json(filepath="data/interim/freight_forecast_30d.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 5 vessels, 10 routes matching the locked Alpha scope
    routes = [f"R-{i:02d}" for i in range(1, 11)]
    vessels = [f"V-00{i}" for i in range(1, 6)]
    base_date = date.today()
    
    records = []
    for day_offset in range(1, 31):
        current_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for route in routes:
            for vessel in vessels:
                base_rate = round(random.uniform(15, 25), 2)
                records.append({
                    "date_index": current_date,
                    "vessel_id": vessel,
                    "route_id": route,
                    "p10_rate": round(base_rate * 0.8, 2),
                    "p50_rate": base_rate,
                    "p90_rate": round(base_rate * 1.3, 2)
                })
    
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2)

if __name__ == "__main__":
    generate_dummy_json()