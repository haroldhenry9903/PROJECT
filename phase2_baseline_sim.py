import pandas as pd
import numpy as np

class Battery:
    def __init__(self, capacity_kwh=10.0, max_power_kw=5.0, soc_min=0.15, soc_max=1.0, efficiency_rt=0.95):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.soc_min = soc_min
        self.soc_max = soc_max
        # Efficiency factors
        self.eff_chg = np.sqrt(efficiency_rt)
        self.eff_dis = np.sqrt(efficiency_rt)
        
        # Initial SoC set to soc_min
        self.soc = soc_min

    def update_soc(self, p_request_kw: float) -> float:
        """
        Calculates the battery action based on requested power.
        Positive p_request_kw means discharging (providing power to load).
        Negative p_request_kw means charging (absorbing surplus PV).
        Returns the actual power provided/absorbed by the battery.
        """
        # Calculate max energy limits based on current SoC
        max_charge_kwh = (self.soc_max - self.soc) * self.capacity_kwh
        max_discharge_kwh = (self.soc - self.soc_min) * self.capacity_kwh

        if p_request_kw > 0.0:  # Discharge attempt
            # Limit by max inverter power
            p_discharge = min(p_request_kw, self.max_power_kw)
            
            # Equation 3.10 refinement: dividing discharging power by discharge efficiency factor.
            # This represents the energy practically drawn from the battery.
            e_discharge_needed = p_discharge / self.eff_dis
            
            # Clamp to max discharge capacity
            e_discharge_actual = min(e_discharge_needed, max_discharge_kwh)
            
            # The useful power delivered to the grid/load after efficiency losses
            p_actual_delivered = e_discharge_actual * self.eff_dis
            
            # Update SoC
            self.soc -= e_discharge_actual / self.capacity_kwh
            return p_actual_delivered
            
        elif p_request_kw < 0.0:  # Charge attempt
            p_charge = min(abs(p_request_kw), self.max_power_kw)
            
            # Equation 3.10 refinement: multiplying charging power by efficiency factor.
            # This represents the actual energy stored in the battery.
            e_charge_effective = p_charge * self.eff_chg
            
            # Clamp to max charge capacity
            e_charge_actual = min(e_charge_effective, max_charge_kwh)
            
            # The power drawn from the source (PV/Grid) before efficiency losses
            p_actual_absorbed = e_charge_actual / self.eff_chg
            
            # Update SoC
            self.soc += e_charge_actual / self.capacity_kwh
            return -p_actual_absorbed
            
        else:
            return 0.0


def generate_synthetic_load(index: pd.DatetimeIndex) -> pd.Series:
    """
    Generates a synthetic residential load profile.
    - Low at night
    - Morning peak (06:00 - 08:00)
    - Evening peak (18:00 - 22:00)
    - Total daily consumption around 15-20 kWh
    - Added stochastic noise
    """
    hours = index.hour
    
    # Base load profile mapping per hour (approximate relative shape)
    # We will scale this base shape and add noise.
    base_shape = np.zeros(len(index))
    
    # Simple predefined shape rules
    # Night time (00:00 - 05:00): Heavy EV Charging / AC
    mask_night = (hours >= 0) & (hours < 6)
    base_shape[mask_night] = 1.8
    
    # Morning peak (06:00 - 08:00): Ramping up
    mask_morning = (hours >= 6) & (hours <= 8)
    base_shape[mask_morning] = 0.8
    
    # Mid-day (09:00 - 17:00): Moderate consumption
    mask_day = (hours >= 9) & (hours <= 17)
    base_shape[mask_day] = 0.5
    
    # Evening peak (18:00 - 22:00): High consumption
    mask_evening = (hours >= 18) & (hours <= 22)
    base_shape[mask_evening] = 1.2
    
    # Late night (23:00): Winding down
    mask_late = (hours == 23)
    base_shape[mask_late] = 0.5
    
    # Add day-by-day stochastic noise (normal distribution)
    # Adding a random scaling factor per day to keep the overall shape but vary intensity
    days = index.normalize()
    unique_days = days.unique()
    
    load = pd.Series(base_shape, index=index)
    
    # Scale base shape to reach ~15-20 kWh daily average
    # The sum of base_shape per day is roughly 14.3. 
    # We will multiply by a scaling factor centered around 1.2
    for day in unique_days:
        day_mask = days == day
        daily_variation = np.random.normal(loc=1.2, scale=0.15)
        # Add hourly noise
        hourly_noise = np.random.normal(loc=1.0, scale=0.1, size=day_mask.sum())
        
        load.loc[day_mask] = load.loc[day_mask] * daily_variation * hourly_noise
        
    # Ensure no negative loads
    load = load.clip(lower=0.1)
    
    return load

def get_tariff_rate(hour: int) -> float:
    """
    Returns Sarawak Energy Tariff C3 rate based on the hour.
    Peak (07:00-22:00): 0.245 RM/kWh
    Off-Peak (22:00-07:00): 0.139 RM/kWh
    """
    if 7 <= hour < 22:
        return 0.245
    else:
        return 0.139

def main():
    input_filepath = 'sarawak_weather_processed.csv'
    output_filepath = 'baseline_simulation_results.csv'
    
    print(f"Loading Phase 1 synthesized data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print("Generating synthetic residential load profile...")
    df['Load_kW'] = generate_synthetic_load(df.index)
    
    print("Initializing Battery and storing simulation results...")
    battery = Battery()
    
    # Prepare lists to store our hourly results
    soc_history = []
    grid_import_history = []
    hourly_cost_history = []
    battery_action_history = [] # For debugging/viz tracking
    
    print("Running baseline simulation hour-by-hour...")
    for dt, row in df.iterrows():
        p_load = row['Load_kW']
        p_pv = row['Target_Power_kW']
        
        # Calculate net power
        # Positive P_net means load is higher than PV (Deficit) -> Battery should discharge
        # Negative P_net means PV is higher than load (Surplus) -> Battery should charge
        p_net = p_load - p_pv
        
        # P_batt is the actual power supplied (+) or absorbed (-) by the battery
        p_batt = battery.update_soc(p_request_kw=p_net)
        
        # The remaining power must come from the grid.
        # Grid import is positive. Grid export (if any) is negative.
        # P_grid = P_net - P_batt
        # Example 1 (Deficit): P_load = 2, P_pv = 0 -> P_net = 2. Battery supplies P_batt = 1. Grid = 2 - 1 = 1 kW import.
        # Example 2 (Surplus): P_load = 1, P_pv = 3 -> P_net = -2. Battery absorbs P_batt = -2. Grid = -2 - (-2) = 0 kW.
        p_grid = p_net - p_batt
        
        # If the battery cannot absorb all surplus, P_grid might be negative (exporting).
        # We will assume zero compensation for exporting back to the grid for this baseline,
        # so grid costs are only associated with importing (P_grid > 0).
        grid_import_kw = max(0.0, p_grid)
        
        # Calculate cost
        tariff_rate = get_tariff_rate(dt.hour)
        hourly_cost_rm = grid_import_kw * tariff_rate
        
        # Store tracking info
        soc_history.append(battery.soc)
        battery_action_history.append(p_batt)
        grid_import_history.append(grid_import_kw)
        hourly_cost_history.append(hourly_cost_rm)
        
    print("Appending results to dataframe...")
    df['SoC'] = soc_history
    df['Batt_Action_kW'] = battery_action_history
    df['Grid_Import_kW'] = grid_import_history
    df['Hourly_Cost_RM'] = hourly_cost_history
    
    print(f"Saving baseline simulation dataframe to {output_filepath}...")
    df.to_csv(output_filepath)
    
    print("Phase 2: Baseline Environment Simulation completed successfully!")

if __name__ == "__main__":
    main()
