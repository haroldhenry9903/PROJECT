import pandas as pd
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Step 1: Data Preprocessing
    1. Load data
    2. Parse datetime and set as index
    3. Localize to 'Asia/Kuching'
    4. Resample to 1H
    5. Interpolate (limit=2) and ffill
    6. Clip specific columns to strictly >= 0.0
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Parsing datetime and setting index...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print("Localizing timezone to Asia/Kuching...")
    # Handle if already timezone-aware or not
    if df.index.tz is None:
        df = df.tz_localize('Asia/Kuching')
    else:
        df = df.tz_convert('Asia/Kuching')
        
    print("Resampling to 1H frequency...")
    # This exposes missing hourly rows
    df = df.resample('1H').asfreq()
    
    print("Handling missing data...")
    # Interpolate linearly for gaps up to 2 hours
    df = df.interpolate(method='linear', limit=2)
    # Forward-fill for any gaps larger than 2 hours
    df = df.ffill()
    
    print("Enforcing physical limits (>= 0.0)...")
    cols_to_clip = ['precip', 'solarradiation', 'ghiradiation', 'dniradiation', 'difradiation']
    for col in cols_to_clip:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
            
    return df

def simulate_pv_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Physics-Based Synthesis (pvlib)
    1. Location: Kuching, Sarawak (Lat 1.55, Lon 110.36, TZ 'Asia/Kuching')
    2. PV System: 5 kWp, mono-Si, 5 deg tilt, 180 azimuth, -0.4% / C temp coefficient
    3. Calculate hourly power (kW)
    4. Append Target_Power_kW
    """
    print("Defining physical location...")
    kuching = Location(latitude=1.55, longitude=110.36, tz='Asia/Kuching', name='Kuching')
    
    print("Defining PV System parameters...")
    # 5 kWp = 5000 W capacity
    # Temperature coefficient of -0.4% / °C translates to gamma_pdc = -0.004
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = PVSystem(
        surface_tilt=5,
        surface_azimuth=180,
        module_parameters={'pdc0': 5000, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 5000}, # simple PVWatts inverter expects sizing in pdc0
        temperature_model_parameters=temperature_model_parameters
    )
    
    print("Running pvlib ModelChain...")
    # Using 'physical' for aoi and 'no_loss' for spectral since we lack specific material/spectral constraints
    mc = ModelChain(system, kuching, aoi_model='physical', spectral_model='no_loss')
    
    # pvlib expects a weather DataFrame with specific column names:
    # ghi, dni, dhi, temp_air, wind_speed
    weather = pd.DataFrame(index=df.index)
    weather['ghi'] = df['ghiradiation']
    weather['dni'] = df['dniradiation']
    weather['dhi'] = df['difradiation']
    weather['temp_air'] = df['temp']
    
    # Default wind speed if unavailable in dataset, needed for temperature modeling
    weather['wind_speed'] = df.get('windspeed', 1.0)
    
    # Run the model with our cleaned weather data
    mc.run_model(weather)
    
    print("Synthesizing final target variable (Target_Power_kW)...")
    # mc.results.ac gives alternating current output in Watts, we want kW
    target_power_kw = mc.results.ac / 1000.0
    
    # Handle minor negative values at night due to inverter consumption and NaN anomalies
    target_power_kw = target_power_kw.clip(lower=0.0).fillna(0.0)
    
    # Append to our original DataFrame
    df['Target_Power_kW'] = target_power_kw
    
    return df

def main():
    input_filepath = 'sarawak_weather_data.csv'
    output_filepath = 'sarawak_weather_processed.csv'
    
    # Execute Step 1
    df_clean = load_and_preprocess_data(input_filepath)
    
    # Execute Step 2
    df_final = simulate_pv_generation(df_clean)
    
    # Save the processed dataset
    print(f"Saving processed dataframe to {output_filepath}...")
    df_final.to_csv(output_filepath)
    
    print("Phase 1: Digital Ground Truth generation completed successfully!")

if __name__ == "__main__":
    main()
