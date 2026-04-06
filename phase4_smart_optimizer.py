import pandas as pd
import numpy as np
import joblib

class Battery:
    def __init__(self, capacity_kwh=10.0, max_charge_kw=5.0, max_discharge_kw=5.0, soc_min=0.15, soc_max=1.0, efficiency_rt=0.95):
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.eff_chg = np.sqrt(efficiency_rt)
        self.eff_dis = np.sqrt(efficiency_rt)
        self.soc = soc_min

    def update_soc(self, p_request_kw: float) -> float:
        max_charge_kwh = (self.soc_max - self.soc) * self.capacity_kwh
        max_discharge_kwh = (self.soc - self.soc_min) * self.capacity_kwh

        if p_request_kw > 0.0:  # Discharge attempt
            p_discharge = min(p_request_kw, self.max_discharge_kw)
            e_discharge_needed = p_discharge / self.eff_dis
            e_discharge_actual = min(e_discharge_needed, max_discharge_kwh)
            p_actual_delivered = e_discharge_actual * self.eff_dis
            self.soc -= e_discharge_actual / self.capacity_kwh
            return p_actual_delivered
            
        elif p_request_kw < 0.0:  # Charge attempt
            p_charge = min(abs(p_request_kw), self.max_charge_kw)
            e_charge_effective = p_charge * self.eff_chg
            e_charge_actual = min(e_charge_effective, max_charge_kwh)
            p_actual_absorbed = e_charge_actual / self.eff_chg
            self.soc += e_charge_actual / self.capacity_kwh
            return -p_actual_absorbed
            
        else:
            return 0.0

def get_tariff_rate(hour: int) -> float:
    """Peak Rate: 0.245 RM/kWh (07:00 to 22:00), Off-Peak: 0.139 RM/kWh."""
    if 7 <= hour < 22:
        return 0.245
    else:
        return 0.139

def main():
    # Load dataset from Phase 2 which already contains the generated Load_kW
    data_path = 'baseline_simulation_results.csv'
    model_path = 'rf_forecast_model.joblib'
    output_path = 'smart_simulation_results.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print(f"Loading trained AI model from {model_path}...")
    rf_model = joblib.load(model_path)
    
    # Ensure temporal features exist for the model
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month
        
    features = ['ghiradiation', 'temp', 'humidity', 'precip', 'cloudcover', 'hour', 'month']
    
    print("Initializing Smart Battery Controller...")
    battery = Battery()
    
    conservation_mode = False
    e_crit = 18.0  # Critical energy threshold in kWh   
    
    # Lists to store results
    soc_history = []
    grid_import_history = []
    hourly_cost_history = []
    conservation_mode_history = []
    
    print("Running Smart Optimizer Simulation...")
    for i, (dt, row) in enumerate(df.iterrows()):
        
        # 1. Midnight Forecast Step
        if dt.hour == 0:
            # Extract the next 24 hours of features safely
            next_24_rows = df.iloc[i : i+24]
            X_24h = next_24_rows[features]
            
            if len(X_24h) < 24:
                conservation_mode = False
            else:
                # Predict the solar generation for the upcoming day
                predictions = rf_model.predict(X_24h)
                e_pred = predictions.sum()
                
                # Mode Selection
                if e_pred < e_crit:
                    conservation_mode = True
                else:
                    conservation_mode = False
                
        # 2. Dynamic Discharge Limit Enforcement
        if conservation_mode and (0 <= dt.hour <= 6):
            battery.max_discharge_kw = 0.0  # Hoarding energy for morning peak
        else:
            battery.max_discharge_kw = 5.0  # Standard continuous rating
            
        # 3. Energy Flow Calculation
        p_load = row['Load_kW']
        p_pv = row['Target_Power_kW']
        
        p_net = p_load - p_pv
        p_batt = battery.update_soc(p_request_kw=p_net)
        
        p_grid = p_net - p_batt
        grid_import_kw = max(0.0, p_grid)
        
        # 4. Hourly Cost Calculation using Target Tariff C3
        tariff_rate = get_tariff_rate(dt.hour)
        hourly_cost_rm = grid_import_kw * tariff_rate
        
        # 5. Store Data
        soc_history.append(battery.soc)
        grid_import_history.append(grid_import_kw)
        hourly_cost_history.append(hourly_cost_rm)
        conservation_mode_history.append(conservation_mode)
        
    print("Appending new results to dataframe...")
    df['SoC'] = soc_history
    df['Grid_Import_kW'] = grid_import_history
    df['Hourly_Cost_RM'] = hourly_cost_history
    df['Conservation_Mode_Active'] = conservation_mode_history
    
    print(f"Saving Smart Simulation Results to {output_path}...")
    df.to_csv(output_path)
    print("Phase 4: Smart Optimizer Simulation completed successfully!")

if __name__ == '__main__':
    main()
