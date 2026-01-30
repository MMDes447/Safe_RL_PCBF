import os
import gc
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from mpc_filter import safety_MILP_two_stage
from steps.fast_step import RADstep_fast, D_lookup, D_vert_lookup, L
from steps.RAD_STEP import RADstep

# ============================================================================
# RADIANCE FILE PATHS (for RADstep)
# ============================================================================
file_path = "Env/Elec_light/ecoo1.rad"
file_path2 = "Env/Elec_light/ecoo2.rad"
file_path3 = "Env/Elec_light/ecoo3.rad"
file_path4 = "Env/Elec_light/ecoo4.rad"
file_path5 = "Env/Elec_light/GRBID940.rad"
file_path6 = "Env/Elec_light/GRBID927.rad"

File_paths = [
    (file_path, 16),
    (file_path2, 16),
    (file_path3, 16),
    (file_path4, 16),
    (file_path5, 16),
    (file_path6, 16)
]
res_path = "ill_read.dat"


USE_FAST_STEP = True


MPC_HORIZON = 2
MPC_DELTA_MAX = 1
MPC_W_BLIND = 0.5
MPC_W_ENERGY = 0.0  # No energy optimization

#
num_days = 365
timesteps_per_day = 96
log_interval = 1
save_interval = 7

results_dir = f"results_MPC_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/plots", exist_ok=True)


mpc_actions_history = []
mpc_blinds_history = []
dgp_history = []
vert_illum_history = []
timestep_energies = []
timestep_illuminances_per_sensor = [[] for _ in range(10)]
timestep_illuminances_avg = []
timestep_daylight_avg = []
slack_upper_history = []
slack_lower_history = []

prev_u = np.ones(6) * 0.5
prev_x = np.ones(10) * 700


num_years = 1  # MPC doesn't learn, 1 year is enough for evaluation
print(f"Starting MPC-only simulation for {num_years} years ({num_years * num_days} days)")
print(f"Step function: {'FAST (lookup)' if USE_FAST_STEP else 'FULL (Radiance)'}")
print(f"MPC Horizon: {MPC_HORIZON}")
print(f"Energy weight: {MPC_W_ENERGY} (disabled)")
print(f"Safety bounds: [500, 800] lux")
print("=" * 60)

for year in range(num_years):
    print(f"\n{'#'*60}")
    print(f"# YEAR {year + 1}/{num_years}")
    print(f"{'#'*60}")
    
    for day in range(num_days):
        global_day = year * num_days + day
        day_energy = 0
        start_timestep = global_day * timesteps_per_day
        
        for timestep_in_day in range(timesteps_per_day):
            global_timestep = start_timestep + timestep_in_day
            lookup_timestep = global_timestep % 35040
            
            verbose = (global_timestep < 5)
            
            # ================================================================
            # MPC CONTROLLER (NO RL - USE NOMINAL INPUT)
            # ================================================================
            # Nominal action: mid-range dimming, blinds fully open
            nominal_continuous = np.ones(6) * 0.5
            nominal_blind = 0  # Fully open
            
            mpc_continuous, mpc_blind, diagnostics = safety_MILP_two_stage(
                a_RL=nominal_continuous,
                b_RL=nominal_blind,
                x_prev=prev_x,
                u_prev=prev_u,
                t=lookup_timestep,
                D=D_lookup,
                D_vert=D_vert_lookup,
                N=MPC_HORIZON,
                delta_max=MPC_DELTA_MAX,
                w_blind=MPC_W_BLIND,
                w_energy=MPC_W_ENERGY,  # No energy optimization
                verbose=verbose
            )
            
            # Track actions
            mpc_actions_history.append(mpc_continuous.copy())
            mpc_blinds_history.append(mpc_blind)
            
            # Track slack
            if diagnostics is not None:
                slack_upper_history.append(diagnostics.get('slack_upper', 0))
                slack_lower_history.append(diagnostics.get('slack_lower', 0))
            else:
                slack_upper_history.append(0)
                slack_lower_history.append(0)
            
            prev_u = mpc_continuous.copy()
            
            # ================================================================
            # STEP FUNCTION
            # ================================================================
            if USE_FAST_STEP:
                next_state, info, _, done = RADstep_fast(
                    mpc_continuous,
                    mpc_blind,
                    lookup_timestep,
                    mpc_correction=0.0
                )
            else:
                next_state, info, _, done = RADstep(
                    DIM_VEC=mpc_continuous,
                    a_id=mpc_blind,
                    File_paths=File_paths,
                    res_path=res_path,
                    step=lookup_timestep,
                    mpc_correction=0.0
                )
            
            prev_x = np.array(info["tot_illuminance"])
            
            vert_illum = np.array(info["vertical_illuminance"])
            dgp = 6.22e-5 * vert_illum + 0.184
            dgp_history.append(np.max(dgp))
            vert_illum_history.append(np.max(vert_illum))
            
            timestep_energies.append(info["power_consumed"])
            
            tot_illum = info["tot_illuminance"]
            for sensor_idx in range(10):
                timestep_illuminances_per_sensor[sensor_idx].append(tot_illum[sensor_idx])
            timestep_illuminances_avg.append(np.mean(tot_illum))
            timestep_daylight_avg.append(np.mean(info["daylight_illuminance"]))
            
            day_energy += info["power_consumed"]
        
        if (global_day + 1) % log_interval == 0:
            day_illum = timestep_illuminances_avg[-96:]
            day_dgp = dgp_history[-96:]
            day_vert = vert_illum_history[-96:]
            day_slack_upper = slack_upper_history[-96:]
            day_slack_lower = slack_lower_history[-96:]
            
            print(f"\n{'='*60}")
            print(f"Day {global_day+1} (Year {year+1})")
            print(f"{'='*60}")
            
            print(f"\n--- MPC SLACK USAGE ---")
            print(f"  Avg upper slack: {np.mean(day_slack_upper):.4f}")
            print(f"  Avg lower slack: {np.mean(day_slack_lower):.4f}")
            print(f"  Max upper slack: {np.max(day_slack_upper):.4f}")
            print(f"  Max lower slack: {np.max(day_slack_lower):.4f}")
            
            print(f"\n--- SAFETY ---")
            in_range = sum(1 for x in day_illum if 500 <= x <= 800)
            below = sum(1 for x in day_illum if x < 500)
            above = sum(1 for x in day_illum if x > 800)
            print(f"  In range [500-800]: {100*in_range/96:.1f}%")
            print(f"  Below 500: {100*below/96:.1f}%")
            print(f"  Above 800: {100*above/96:.1f}%")
            print(f"  Avg illuminance: {np.mean(day_illum):.1f} lux")
            print(f"  Min illuminance: {np.min(day_illum):.1f} lux")
            print(f"  Max illuminance: {np.max(day_illum):.1f} lux")
            
            print(f"\n--- GLARE ---")
            print(f"  Max DGP: {np.max(day_dgp):.3f}")
            print(f"  Avg DGP: {np.mean(day_dgp):.3f}")
            print(f"  Max vertical illum: {np.max(day_vert):.0f} lux")
            glare_ok = sum(1 for x in day_dgp if x < 0.35)
            print(f"  DGP < 0.35: {100*glare_ok/96:.1f}%")
            
            print(f"\n--- ENERGY (NOT OPTIMIZED) ---")
            print(f"  Daily energy: {day_energy:.2f} Wh")
        
        if (global_day + 1) % save_interval == 0:
            results_path = f"{results_dir}/mpc_results_day{global_day+1}.npz"
            np.savez(
                results_path,
                mpc_actions=np.array(mpc_actions_history),
                mpc_blinds=np.array(mpc_blinds_history),
                illuminances=np.array(timestep_illuminances_avg),
                dgp=np.array(dgp_history),
                energies=np.array(timestep_energies),
            )
            print(f"\n  Results saved to {results_path}")
            
            # ================================================================
            # PLOTTING
            # ================================================================
            fig = plt.figure(figsize=(20, 20))
            window = 96
            
            plt.subplot(5, 3, 1)
            mpc_avg = [np.mean(a) for a in mpc_actions_history]
            plt.plot(mpc_avg, alpha=0.7, linewidth=0.5, color='orange')
            plt.title('MPC Dimming (Average)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            
            plt.subplot(5, 3, 2)
            plt.plot(mpc_blinds_history, alpha=0.7, linewidth=0.5, color='orange')
            plt.title('MPC Blind Position')
            plt.xlabel('Timestep')
            plt.ylabel('Blind Position')
            plt.ylim(-0.5, 16.5)
            plt.grid(True)
            
            plt.subplot(5, 3, 3)
            mpc_blind_counts = [mpc_blinds_history.count(i) for i in range(17)]
            plt.bar(range(17), mpc_blind_counts, alpha=0.7, color='orange')
            plt.title('Blind Action Distribution')
            plt.xlabel('Blind Position')
            plt.ylabel('Count')
            plt.grid(True)
            
            plt.subplot(5, 3, 4)
            plt.plot(timestep_illuminances_avg, linewidth=0.5)
            plt.axhline(y=500, color='r', linestyle='--', label='Min (500)')
            plt.axhline(y=800, color='r', linestyle='--', label='Max (800)')
            plt.title('Total Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(5, 3, 5)
            compliance = []
            for i in range(len(timestep_illuminances_avg)):
                start = max(0, i - window + 1)
                in_range = sum(1 for x in timestep_illuminances_avg[start:i+1] if 500 <= x <= 800)
                compliance.append(100 * in_range / (i - start + 1))
            plt.plot(compliance, linewidth=0.5, color='green')
            plt.title('Illuminance Compliance (Rolling)')
            plt.xlabel('Timestep')
            plt.ylabel('%')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(5, 3, 6)
            for i in range(10):
                plt.plot(timestep_illuminances_per_sensor[i], label=f'S{i+1}', alpha=0.5, linewidth=0.3)
            plt.axhline(y=500, color='r', linestyle='--')
            plt.axhline(y=800, color='r', linestyle='--')
            plt.title('Per-Sensor Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(5, 3, 7)
            plt.plot(dgp_history, linewidth=0.5, color='purple')
            plt.axhline(y=0.35, color='r', linestyle='--', label='Imperceptible (0.35)')
            plt.axhline(y=0.40, color='orange', linestyle='--', label='Perceptible (0.40)')
            plt.title('Daylight Glare Probability (DGP)')
            plt.xlabel('Timestep')
            plt.ylabel('DGP')
            plt.ylim(0, 0.6)
            plt.legend()
            plt.grid(True)
            
            plt.subplot(5, 3, 8)
            rolling_glare_ok = []
            for i in range(len(dgp_history)):
                start = max(0, i - window + 1)
                ok = sum(1 for x in dgp_history[start:i+1] if x < 0.35)
                rolling_glare_ok.append(100 * ok / (i - start + 1))
            plt.plot(rolling_glare_ok, linewidth=0.5, color='purple')
            plt.title('Glare Compliance (DGP < 0.35)')
            plt.xlabel('Timestep')
            plt.ylabel('%')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(5, 3, 9)
            plt.plot(vert_illum_history, linewidth=0.5, color='purple')
            plt.axhline(y=2669, color='r', linestyle='--', label='DGP=0.35')
            plt.axhline(y=3473, color='orange', linestyle='--', label='DGP=0.40')
            plt.title('Vertical Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(5, 3, 10)
            plt.plot(slack_upper_history, label='Upper', alpha=0.7, linewidth=0.5)
            plt.plot(slack_lower_history, label='Lower', alpha=0.7, linewidth=0.5)
            plt.title('MPC Slack Usage')
            plt.xlabel('Timestep')
            plt.ylabel('Slack')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(5, 3, 11)
            plt.plot(timestep_daylight_avg, linewidth=0.5, color='gold')
            plt.title('Daylight Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.grid(True)
            
            plt.subplot(5, 3, 12)
            plt.plot(timestep_energies, linewidth=0.5)
            plt.title('Energy Consumption (NOT optimized)')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(5, 3, 13)
            for i in range(6):
                mpc_light = [a[i] for a in mpc_actions_history]
                plt.plot(mpc_light, label=f'L{i+1}', alpha=0.5, linewidth=0.3)
            plt.title('Per-Light Dimming')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming')
            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(5, 3, 14)
            rolling_energy = []
            for i in range(len(timestep_energies)):
                start = max(0, i - window + 1)
                rolling_energy.append(np.mean(timestep_energies[start:i+1]))
            plt.plot(rolling_energy, linewidth=0.5, color='orange')
            plt.title('Rolling Avg Energy')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(5, 3, 15)
            # Histogram of illuminance values
            plt.hist(timestep_illuminances_avg, bins=50, alpha=0.7)
            plt.axvline(x=500, color='r', linestyle='--', label='Min (500)')
            plt.axvline(x=800, color='r', linestyle='--', label='Max (800)')
            plt.title('Illuminance Distribution')
            plt.xlabel('Lux')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/plots/detailed_day{global_day+1}.png", dpi=150)
            plt.close()
    
    
    print(f"\n{'='*60}")
    print(f"YEAR {year + 1} COMPLETED")
    print(f"{'='*60}")
    
    # Year summary
    year_illum = timestep_illuminances_avg
    year_dgp = dgp_history
    year_energy = timestep_energies
    
    in_range = sum(1 for x in year_illum if 500 <= x <= 800)
    glare_ok = sum(1 for x in year_dgp if x < 0.35)
    
    print(f"\n--- YEAR {year+1} SUMMARY ---")
    print(f"  Illuminance compliance: {100*in_range/len(year_illum):.1f}%")
    print(f"  Glare compliance: {100*glare_ok/len(year_dgp):.1f}%")
    print(f"  Total energy: {sum(year_energy):.0f} Wh")
    print(f"  Avg energy per day: {sum(year_energy)/num_days:.0f} Wh")
    
    # Clear for next year
    mpc_actions_history = []
    mpc_blinds_history = []
    dgp_history = []
    vert_illum_history = []
    timestep_energies = []
    timestep_illuminances_per_sensor = [[] for _ in range(10)]
    timestep_illuminances_avg = []
    timestep_daylight_avg = []
    slack_upper_history = []
    slack_lower_history = []
    
    plt.close('all')
    gc.collect()

print("\n" + "=" * 60)
print("SIMULATION COMPLETED!")
print("=" * 60)
print(f"Total years: {num_years}")
print(f"Total days: {num_years * num_days}")
print(f"\nResults saved to {results_dir}")