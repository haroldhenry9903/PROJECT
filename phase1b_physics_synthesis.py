"""Phase 1B: pvlib model chain that turns cleaned hourly weather into `Target_Power_kW`.

Refactored so the location (latitude/longitude) is a parameter, not a literal.
This is the only place `pvlib.Location` is constructed in the pipeline, so the
multi-region phase-1 helper (`phase1_split_multi_region.py`) reuses it for every
Sarawak site without copy/pasting the model-chain configuration.

Backward compatibility:
  - Running `python phase1b_physics_synthesis.py` from the CLI keeps the old
    single-site (Kuching) behaviour and reads/writes the legacy paths from
    `experiment_config.json#paths.phase1_processed_csv`.
"""
from __future__ import annotations

import pandas as pd
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from repo_paths import repo_root, resolve_repo_path


def build_pv_system() -> PVSystem:
    """Return the canonical 5 kWp Sarawak rooftop system shared by every region.

    Tilt/azimuth/temperature-coefficient are deliberately identical across
    regions; only the geographic Location (lat/lon) changes. This isolates the
    region effect to *meteorology*, not hardware.
    """
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    return PVSystem(
        surface_tilt=5,
        surface_azimuth=180,
        module_parameters={"pdc0": 5000, "gamma_pdc": -0.004},
        inverter_parameters={"pdc0": 5000},
        temperature_model_parameters=temperature_model_parameters,
    )


def simulate_pv_for_location(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    region_name: str = "kuching",
    tz: str = "Asia/Kuching",
) -> pd.DataFrame:
    """Append `Target_Power_kW` to an hourly weather dataframe via pvlib.

    Inputs:
      df              -- cleaned hourly dataframe; must contain the columns
                         ['ghiradiation', 'dniradiation', 'difradiation', 'temp'].
                         A `windspeed` column is used if present, else defaults
                         to a constant 1.0 m/s (matches the legacy script).
      latitude, longitude -- site coordinates (decimal degrees, WGS84).
      region_name     -- friendly label for logs; does not affect the model.
      tz              -- pvlib Location timezone. For Sarawak this is always
                         'Asia/Kuching' (single time zone for the whole state).

    Side-effects:
      The function does NOT mutate `df`. It returns a NEW dataframe with the
      same columns plus `Target_Power_kW` (kW; clipped to >= 0, NaNs -> 0).
    """
    location = Location(latitude=latitude, longitude=longitude, tz=tz, name=region_name)
    system = build_pv_system()
    mc = ModelChain(system, location, aoi_model="physical", spectral_model="no_loss")

    weather = pd.DataFrame(index=df.index)
    weather["ghi"] = df["ghiradiation"]
    weather["dni"] = df["dniradiation"]
    weather["dhi"] = df["difradiation"]
    weather["temp_air"] = df["temp"]
    weather["wind_speed"] = df.get("windspeed", 1.0)

    mc.run_model(weather)

    target_power_kw = (mc.results.ac / 1000.0).clip(lower=0.0).fillna(0.0)

    out = df.copy()
    out["Target_Power_kW"] = target_power_kw
    return out


def simulate_pv_generation(input_file: str, output_file: str,
                            *, latitude: float = 1.55, longitude: float = 110.36,
                            region_name: str = "kuching") -> None:
    """Legacy single-site entry point.

    Reads the cleaned CSV at `input_file`, runs `simulate_pv_for_location`
    with the supplied coordinates, and writes the result to `output_file`
    with a tz-naive index (pipeline convention).
    """
    print(f"Loading cleaned dataset from {input_file}...")
    df = pd.read_csv(input_file)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kuching")
    df.set_index("datetime", inplace=True)

    print(
        f"Running pvlib model chain for region '{region_name}' "
        f"at (lat={latitude}, lon={longitude})..."
    )
    out = simulate_pv_for_location(
        df,
        latitude=latitude,
        longitude=longitude,
        region_name=region_name,
    )

    print(f"Saving final synthesized ground-truth dataset to {output_file}...")
    out.index = out.index.tz_localize(None)
    out.to_csv(output_file)
    print("Phase 1B: PV Simulation Synthesis completed successfully!")


if __name__ == "__main__":
    _repo = repo_root()
    simulate_pv_generation(
        input_file=str(resolve_repo_path(_repo, "data/processed/sarawak_weather_cleaned.csv")),
        output_file=str(resolve_repo_path(_repo, "data/processed/sarawak_weather_processed.csv")),
        latitude=1.55,
        longitude=110.36,
        region_name="kuching",
    )
