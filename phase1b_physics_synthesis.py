import pandas as pd
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from repo_paths import repo_root, resolve_repo_path


def simulate_pv_generation(input_file: str, output_file: str):
    print(f"Loading cleaned dataset from {input_file}...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Asia/Kuching')
    df.set_index('datetime', inplace=True)
    
    print("Defining physical system configuration for Kuching, Sarawak...")
    kuching = Location(latitude=1.55, longitude=110.36, tz='Asia/Kuching', name='Kuching')
    
    print("Initializing PV System specifications...")
    # Base specifications from Chapter 3
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    system = PVSystem(
        surface_tilt=5,
        surface_azimuth=180,
        module_parameters={'pdc0': 5000, 'gamma_pdc': -0.004}, # 5kWp capacity, -0.4% temp coefficient
        inverter_parameters={'pdc0': 5000},
        temperature_model_parameters=temperature_model_parameters
    )
    
    print("Running physically-validated solar generation model (pvlib)...")
    mc = ModelChain(system, kuching, aoi_model='physical', spectral_model='no_loss')
    
    weather = pd.DataFrame(index=df.index)
    weather['ghi'] = df['ghiradiation']
    weather['dni'] = df['dniradiation']
    weather['dhi'] = df['difradiation']
    weather['temp_air'] = df['temp']
    weather['wind_speed'] = df.get('windspeed', 1.0)
    
    # Calculate output based on input weather variations
    mc.run_model(weather)
    
    print("Extracting Target_Power_kW...")
    target_power_kw = mc.results.ac / 1000.0  # Convert W to kW
    target_power_kw = target_power_kw.clip(lower=0.0).fillna(0.0) # Ensure no negative generation states
    
    df['Target_Power_kW'] = target_power_kw
    
    print(f"Saving final synthesized ground-truth dataset to {output_file}...")
    # Drop timezone requirement for standard CSV compatibility later
    df.index = df.index.tz_localize(None) 
    df.to_csv(output_file)
    print("Phase 1B: PV Simulation Synthesis completed successfully!")

if __name__ == '__main__':
    _repo = repo_root()
    simulate_pv_generation(
        input_file=str(resolve_repo_path(_repo, "data/processed/sarawak_weather_cleaned.csv")),
        output_file=str(resolve_repo_path(_repo, "data/processed/sarawak_weather_processed.csv")),
    )
