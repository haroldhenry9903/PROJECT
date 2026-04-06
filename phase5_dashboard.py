import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from phase2_baseline_sim import Battery as BaselineBattery, get_tariff_rate
from phase4_smart_optimizer import Battery as SmartBattery

# Streamlit Page Config
st.set_page_config(page_title="AI-Driven Solar Battery Optimizer", layout="wide")

# ==========================================
# Data Loading and Caching
# ==========================================
@st.cache_data
def load_data():
    """Load both baseline and smart simulation results."""
    df_base = pd.read_csv('baseline_simulation_results.csv')
    df_base['datetime'] = pd.to_datetime(df_base['datetime'])
    df_base.set_index('datetime', inplace=True)
    
    df_smart = pd.read_csv('smart_simulation_results.csv')
    df_smart['datetime'] = pd.to_datetime(df_smart['datetime'])
    df_smart.set_index('datetime', inplace=True)
    return df_base, df_smart

df_base, df_smart = load_data()

# ==========================================
# Sidebar: User Controls
# ==========================================
st.sidebar.header("Simulation Controls")
st.sidebar.markdown('**Select a date from the Testing Period (Aug 2024 - Jan 2025)**')

# Filter for testing period dates only
test_period = df_smart.loc['2024-08-08':'2025-01-01']
unique_dates = test_period.index.normalize().unique().date

selected_date = st.sidebar.date_input(
    "Simulation Date", 
    value=unique_dates[0],
    min_value=min(unique_dates),
    max_value=max(unique_dates)
)

st.title("☀️ AI-Driven Solar Battery Optimizer (Digital Twin)")
st.markdown("""
This dashboard compares a standard reactive battery controller against our AI-Driven Smart Controller. 
The AI predicts low-solar days 24 hours in advance to activate **Conservation Mode**, avoiding nighttime discharge and hoarding energy for the expensive morning Tariff C3 peak.
""")

# Setup Tabs
tab1, tab2, tab3 = st.tabs(["Daily Operational View", "Global Performance Metrics", "Hardware Sensitivity Analysis"])

# ==========================================
# TAB 1: Daily Operational View
# ==========================================
with tab1:
    st.header(f"Daily View: {selected_date}")
    
    # Filter data for the selected day
    date_str = str(selected_date)
    try:
        daily_smart = df_smart.loc[date_str]
        daily_base = df_base.loc[date_str]
        
        if daily_smart.empty:
            st.warning("No data available for the selected date. Please choose a date within the testing range.")
        else:
            # 1. Operational Mode Metric Box
            mode_active = daily_smart['Conservation_Mode_Active'].iloc[0]
            mode_color = "red" if mode_active else "green"
            mode_text = "⚠️ CONSERVATION MODE (Hoarding active)" if mode_active else "✅ STANDARD MODE (Self-Consumption)"
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); text-align: center; border: 2px solid {mode_color};">
                <h3 style="margin:0; color:{mode_color};">{mode_text}</h3>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            # Show Load & Solar curves
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Power Demand vs Solar Generation")
                fig_power, ax_power = plt.subplots(figsize=(8, 4))
                ax_power.plot(daily_smart.index.hour, daily_smart['Load_kW'], label='Load (kW)', color='orange', linewidth=2)
                ax_power.plot(daily_smart.index.hour, daily_smart['Target_Power_kW'], label='Actual PV (kW)', color='green', linewidth=2)
                ax_power.axvspan(7, 22, color='red', alpha=0.1, label='Peak Tariff Period')
                ax_power.set_xlabel('Hour of Day')
                ax_power.set_ylabel('Power (kW)')
                ax_power.legend()
                ax_power.grid(True, alpha=0.3)
                st.pyplot(fig_power)
                
            with col2:
                st.subheader("Battery SoC Comparison (AI vs Dumb)")
                fig_soc, ax_soc = plt.subplots(figsize=(8, 4))
                ax_soc.plot(daily_base.index.hour, daily_base['SoC'] * 100, label='Baseline SoC %', color='gray', linestyle='--')
                ax_soc.plot(daily_smart.index.hour, daily_smart['SoC'] * 100, label='Smart AI SoC %', color='blue', linewidth=2)
                ax_soc.axhline(15, color='red', linestyle=':', label='Min PvP (15%)')
                if mode_active:
                    ax_soc.axvspan(0, 6, color='blue', alpha=0.1, label='Hoarding Window (AI)')
                ax_soc.set_xlabel('Hour of Day')
                ax_soc.set_ylabel('State of Charge (%)')
                ax_soc.legend(loc='upper right')
                ax_soc.grid(True, alpha=0.3)
                st.pyplot(fig_soc)
                
            # 2. Daily Cost Comparison
            daily_cost_base = daily_base['Hourly_Cost_RM'].sum()
            daily_cost_smart = daily_smart['Hourly_Cost_RM'].sum()
            savings = daily_cost_base - daily_cost_smart
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Baseline Daily Cost", f"RM {daily_cost_base:.2f}")
            mc2.metric("Smart Daily Cost", f"RM {daily_cost_smart:.2f}", delta=f"RM {-savings:.2f}", delta_color="inverse")
            mc3.metric("Daily Savings", f"RM {savings:.2f}")
            
            # 3. 30-Day SoC Heatmap
            st.divider()
            st.subheader("30-Day State of Charge (SoC) Heatmap (Smart AI)")
            st.markdown("Visualizing how the AI restricts nighttime discharge during low-solar periods to save energy for morning peaks.")
            
            # Extract 15 days before and 15 days after selected date
            selected_dt = pd.to_datetime(selected_date)
            start_dt = selected_dt - pd.Timedelta(days=15)
            end_dt = selected_dt + pd.Timedelta(days=15)
            
            # Filter and pivot
            try:
                heatmap_data = df_smart.loc[str(start_dt.date()):str(end_dt.date())].copy()
                heatmap_data['Date'] = heatmap_data.index.date
                heatmap_data['Hour'] = heatmap_data.index.hour
                
                # Pivot: Date across X, Hour across Y
                pivot_soc = heatmap_data.pivot_table(values='SoC', index='Hour', columns='Date')
                
                fig_heat, ax_heat = plt.subplots(figsize=(12, 6))
                sns.heatmap(pivot_soc * 100, cmap="YlGnBu", ax=ax_heat, cbar_kws={'label': 'SoC %'})
                ax_heat.set_title("SoC Behavior over 30 Days")
                st.pyplot(fig_heat)
            except Exception as e:
                st.error("Could not generate heatmap for this specific date range due to boundary limits.")

    except KeyError:
        st.error("Selected date is missing from the dataset. Please adjust the sidebar filter.")

# ==========================================
# TAB 2: Global Performance Metrics
# ==========================================
with tab2:
    st.header("Overall Testing Period Economics (August 2024 - Jan 2025)")
    
    # Calculate Global metrics over test period only
    base_test = df_base.loc['2024-08-08':'2025-01-01']
    smart_test = df_smart.loc['2024-08-08':'2025-01-01']
    
    total_cost_base = base_test['Hourly_Cost_RM'].sum()
    total_cost_smart = smart_test['Hourly_Cost_RM'].sum()
    total_savings = total_cost_base - total_cost_smart
    pct_savings = (total_savings / total_cost_base) * 100
    
    # Self-Sufficiency Ratio (SSR)
    # Formula: (Sum(P_load) - Sum(P_grid_import)) / Sum(P_load)
    total_load = smart_test['Load_kW'].sum()
    total_import_base = base_test['Grid_Import_kW'].sum()
    total_import_smart = smart_test['Grid_Import_kW'].sum()
    
    ssr_base = ((total_load - total_import_base) / total_load) * 100
    ssr_smart = ((total_load - total_import_smart) / total_load) * 100
    
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Total Baseline Cost", f"RM {total_cost_base:,.2f}")
    gc2.metric("Total AI Optimized Cost", f"RM {total_cost_smart:,.2f}", delta=f"{-pct_savings:.1f}%", delta_color="inverse")
    gc3.metric("Total Project Savings", f"RM {total_savings:,.2f}")
    
    sc1, sc2, _ = st.columns(3)
    sc1.metric("Baseline SSR", f"{ssr_base:.1f}%")
    sc2.metric("AI Optimized SSR", f"{ssr_smart:.1f}%", delta=f"{ssr_smart - ssr_base:.1f}%")

# ==========================================
# TAB 3: Hardware Sensitivity Analysis
# ==========================================
with tab3:
    st.header("Hardware Capacity Sensitivity Analysis (5 kWh to 15 kWh)")
    st.markdown("Proving the robust economic advantage of the AI controller across different residential battery sizes.")

    @st.cache_data
    def run_sensitivity_analysis():
        """
        Runs a simplified version of the logic across varying C_batt sizes to calculate
        the total annual savings of the smart AI vs Baseline.
        """
        capacities = range(5, 16)
        results = []
        
        # We'll use the entire available dataset (Jan 2023 - Jan 2025) roughly localized assuming repetitive 
        # patterns just to calculate a general sensitivity trend to represent "Annualized" equivalents.
        df_sim = df_smart.copy()
        
        # We need a progress bar for UI experience
        progress_bar = st.progress(0)
        
        for idx, cap in enumerate(capacities):
            # Baseline simulation for this cap
            batt_base = BaselineBattery(capacity_kwh=cap, max_power_kw=5.0)
            
            # Smart AI simulation for this cap
            # The AI Battery class is identical physically but receives dynamic overrides
            batt_smart = SmartBattery(capacity_kwh=cap, max_charge_kw=5.0, max_discharge_kw=5.0)
            
            cost_b = 0.0
            cost_s = 0.0
            
            # Run simulation
            for _, row in df_sim.iterrows():
                hr = row['hour']
                net_p = row['Load_kW'] - row['Target_Power_kW']
                t_rate = get_tariff_rate(hr)
                
                # --- Base ---
                p_b = batt_base.update_soc(net_p)
                p_grid_b = max(0.0, net_p - p_b)
                cost_b += p_grid_b * t_rate
                
                # --- Smart ---
                is_conserving = row['Conservation_Mode_Active']
                if is_conserving and 0 <= hr <= 6:
                    batt_smart.max_discharge_kw = 0.0
                else:
                    batt_smart.max_discharge_kw = 5.0
                    
                p_s = batt_smart.update_soc(net_p)
                p_grid_s = max(0.0, net_p - p_s)
                cost_s += p_grid_s * t_rate
                
            savings = cost_b - cost_s
            # Scale roughly to an annualized figure (Dataset is ~2 years => divide by 2)
            annual_savings = savings / 2.0
            
            results.append({'Capacity_kWh': cap, 'Annual_Savings_RM': annual_savings})
            progress_bar.progress((idx + 1) / len(capacities))
            
        progress_bar.empty()
        return pd.DataFrame(results)

    if st.button("Run Sensitivity Simulation (Compute Heavy)"):
        with st.spinner("Simulating physics engine across all hardware configs..."):
            sens_df = run_sensitivity_analysis()
            
            fig_sens, ax_sens = plt.subplots(figsize=(10, 5))
            sns.barplot(data=sens_df, x='Capacity_kWh', y='Annual_Savings_RM', palette='viridis', ax=ax_sens)
            ax_sens.set_xlabel('Battery Capacity (kWh)')
            ax_sens.set_ylabel('Annual Savings (RM/Year)')
            ax_sens.set_title("AI Controller Financial Edge by Battery Size")
            
            for i, v in enumerate(sens_df['Annual_Savings_RM']):
                ax_sens.text(i, v + 5, f"{v:.0f}", color='black', ha='center', fontweight='bold')
                
            st.pyplot(fig_sens)
            st.success("Sensitivity Analysis complete!")
    else:
        st.info("Click the button above to execute the multi-hardware simulation.")
