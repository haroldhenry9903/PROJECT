import numpy as np

class Battery:
    """
    Digital Twin Battery Model
    
    This class simulates the physical behavior of a residential solar battery
    (typically a Lithium Iron Phosphate / LFP battery). 
    It accurately tracks energy flow, enforcing strict hardware limits and efficiency losses.
    
    For non-technical readers: 
    Think of this as a water tank. 
    - Capacity is how much water it holds.
    - Max Charge/Discharge is the width of the pipes (how fast water enters/exits).
    - SoC (State of Charge) is the current water level.
    """
    def __init__(self, capacity_kwh=10.0, max_charge_kw=5.0, max_discharge_kw=5.0, 
                 soc_min=0.15, soc_max=1.0, efficiency_rt=0.95):
        # Physical constraints
        self.capacity_kwh = capacity_kwh        # Tank size (e.g., 10 kWh)
        self.max_charge_kw = max_charge_kw      # Inverter limit: max power going IN
        self.max_discharge_kw = max_discharge_kw # Inverter limit: max power going OUT
        self.soc_min = soc_min                  # Minimal safe level (15%) to prevent battery degradation
        self.soc_max = soc_max                  # Max charge level (100%)
        
        # Round-trip efficiency accounts for chemical and inverter heat losses.
        # If Round-Trip is 95% (0.95), we assume one-way efficiency is sqrt(0.95).
        self.eff_chg = np.sqrt(efficiency_rt)
        self.eff_dis = np.sqrt(efficiency_rt)
        
        # Simulate starting at empty (minimum safe capacity)
        self.soc = soc_min

    def get_energy_kwh(self):
        """Returns the current usable energy inside the battery."""
        return self.soc * self.capacity_kwh

    def update_soc(self, p_request_kw: float) -> float:
        """
        Calculates the battery's reaction to power surplus or deficit.
        
        Logic for Non-Solar Readers:
        1. If p_request_kw is POSITIVE (Deficit): The house needs power. We try to DISCHARGE the battery to help.
        2. If p_request_kw is NEGATIVE (Surplus): The sun is shining bright! We try to CHARGE the battery with excess.
        
        Returns the ACTUAL power supplied or absorbed by the battery after calculating limits.
        """
        max_charge_kwh = (self.soc_max - self.soc) * self.capacity_kwh
        max_discharge_kwh = (self.soc - self.soc_min) * self.capacity_kwh

        if p_request_kw > 0.0:  
            # --- DISCHARGE SCENARIO (House needs power) ---
            # 1. We cannot exceed the inverter's maximum output wire size (max_discharge_kw)
            p_discharge = min(p_request_kw, self.max_discharge_kw)
            
            # 2. Physics: To deliver 1kW to the house, we must pull slightly more than 1kW 
            # from the chemicals due to heat loss (divided by efficiency).
            e_discharge_needed = p_discharge / self.eff_dis
            
            # 3. We can only give what we have. Clamp it if the battery is getting too empty.
            e_discharge_actual = min(e_discharge_needed, max_discharge_kwh)
            
            # The useful power that actually reaches the house
            p_actual_delivered = e_discharge_actual * self.eff_dis
            
            # Update the SoC (reduce the water level)
            self.soc -= e_discharge_actual / self.capacity_kwh
            return p_actual_delivered
            
        elif p_request_kw < 0.0:  
            # --- CHARGE SCENARIO (Excess solar power) ---
            # 1. We cannot exceed the inverter's maximum input wire size
            p_charge = min(abs(p_request_kw), self.max_charge_kw)
            
            # 2. Physics: If we push 1kW in, slightly less than 1kW actually gets stored due to heat.
            e_charge_effective = p_charge * self.eff_chg
            
            # 3. Prevent overcharging (don't overflow the tank)
            e_charge_actual = min(e_charge_effective, max_charge_kwh)
            
            # The actual power absorbed from the solar panel
            p_actual_absorbed = e_charge_actual / self.eff_chg
            
            # Update the SoC (raise the water level)
            self.soc += e_charge_actual / self.capacity_kwh
            
            return -p_actual_absorbed
            
        else:
            return 0.0
