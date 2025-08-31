import pandas as pd
import numpy as np

np.random.seed(42)
n_days = 10

# Client 1: HVAC features
client1_features = pd.DataFrame({
    'avg_temp': np.random.uniform(20, 25, n_days),
    'energy_consumed': np.random.uniform(5, 20, n_days),
    'run_hours': np.random.uniform(6, 12, n_days),
    'cooling_setpoint': np.random.uniform(21, 24, n_days),
    'heating_setpoint': np.random.uniform(18, 22, n_days)
})
client1_features.to_csv("../data/client1_hvac.csv", index=False)

# Client 2: Solar battery features
client2_features = pd.DataFrame({
    'solar_gen': np.random.uniform(0, 15, n_days),
    'battery_charge': np.random.uniform(20, 80, n_days),
    'battery_discharge': np.random.uniform(0, 50, n_days),
    'grid_draw': np.random.uniform(0, 10, n_days),
    'soc': np.random.uniform(10, 90, n_days)
})
client2_features.to_csv("../data/client2_solar.csv", index=False)

# Server labels
labels = pd.DataFrame({
    'solar_for_hvac': client1_features['energy_consumed']*0.5 +
                      client2_features['solar_gen']*0.3 +
                      np.random.normal(0,1,n_days)
})
labels.to_csv("../data/labels.csv", index=False)

print("✅ CSV files saved in data/")
