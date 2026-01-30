import os
import gc
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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

# ============================================================================
# CHOOSE STEP FUNCTION
# ============================================================================
USE_FAST_STEP = True

# ============================================================================
# RULE-BASED CONTROLLER (MEAN - UNIFORM DIMMING)
# ============================================================================
def rule_based_controller_mean(timestep, D_lookup, D_vert_lookup, L):
    """Simple rule-based controller with uniform dimming targeting mean illuminance."""
    TARGET = 600  # Target mean illuminance
    DGP_LIMIT = 0.35
    
    timestep_in_day = timestep % 96
    is_occupied = (8 * 4) <= timestep_in_day < (18 * 4)
    
    if not is_occupied:
        return np.zeros(6), 0
    
    # -------------------------------------------------------------------------
    # BLIND SELECTION: Maximize daylight while respecting glare
    # -------------------------------------------------------------------------
    best_blind = 0
    best_daylight = 0
    
    for blind_id in range(17):
        daylight = D_lookup[timestep, blind_id, :]
        vert_illum = D_vert_lookup[timestep, blind_id, :]
        dgp = 6.22e-5 * vert_illum + 0.184
        max_dgp = np.max(dgp)
        avg_daylight = np.mean(daylight)
        
        if max_dgp < DGP_LIMIT and avg_daylight > best_daylight:
            best_daylight = avg_daylight
            best_blind = blind_id
    
    # Fallback if no config meets glare limit
    if best_daylight == 0:
        for blind_id in range(16, -1, -1):
            vert_illum = D_vert_lookup[timestep, blind_id, :]
            dgp = 6.22e-5 * vert_illum + 0.184
            max_dgp = np.max(dgp)
            if max_dgp < 0.45:
                best_blind = blind_id
                break
        else:
            best_blind = 16
    
    # -------------------------------------------------------------------------
    # DIMMING: Uniform level to reach target MEAN illuminance
    # -------------------------------------------------------------------------
    daylight = D_lookup[timestep, best_blind, :]
    avg_daylight = np.mean(daylight)
    deficit = TARGET - avg_daylight
    
    if deficit <= 0:
        dim_level = np.zeros(6)
    else:
        # Average contribution of all lights to mean illuminance
        avg_light_contribution = np.mean(np.sum(L, axis=1))
        dim_level = np.clip(deficit / avg_light_contribution, 0, 1)
        dim_level = np.ones(6) * dim_level  # Uniform across all 6 lights
    
    return dim_level, best_blind

# ============================================================================
# SIMULATION CONFIG
# ============================================================================
num_days = 365
timesteps_per_day = 96
log_interval = 1
save_interval = 7

results_dir = f"results_RB_mean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/plots", exist_ok=True)

# ============================================================================
# TRACKING VARIABLES
# ============================================================================
rb_actions_history = []
rb_blinds_history = []
dgp_history = []
vert_illum_history = []
timestep_energies = []
timestep_illuminances_per_sensor = [[] for _ in range(10)]
timestep_illuminances_avg = []
timestep_daylight_avg = []

# ============================================================================
# SIMULATION LOOP
# ============================================================================
num_years = 1  # Rule-based doesn't learn, 1 year is enough
print(f"Starting Rule-Based (mean/uniform) simulation for {num_years} years ({num_years * num_days} days)")
print(f"Step function: {'FAST (lookup)' if USE_FAST_STEP else 'FULL (Radiance)'}")
print(f"Target: 600 lux (mean illuminance)")
print(f"Dimming: Uniform across all 6 lights")
print(f"Glare limit: DGP < 0.35")
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
            
            # ================================================================
            # RULE-BASED CONTROLLER (MEAN)
            # ================================================================
            rb_continuous, rb_blind = rule_based_controller_mean(
                lookup_timestep, D_lookup, D_vert_lookup, L
            )
            
            # Track actions
            rb_actions_history.append(rb_continuous.copy())
            rb_blinds_history.append(rb_blind)
            
            # ================================================================
            # STEP FUNCTION
            # ================================================================
            if USE_FAST_STEP:
                next_state, info, _, done = RADstep_fast(
                    rb_continuous,
                    rb_blind,
                    lookup_timestep,
                    mpc_correction=0.0
                )
            else:
                next_state, info, _, done = RADstep(
                    DIM_VEC=rb_continuous,
                    a_id=rb_blind,
                    File_paths=File_paths,
                    res_path=res_path,
                    step=lookup_timestep,
                    mpc_correction=0.0
                )
            
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
            day_blinds = rb_blinds_history[-96:]
            day_dims = rb_actions_history[-96:]
            
            # Per-sensor stats for the day
            day_sensor_illum = [[timestep_illuminances_per_sensor[s][-(96-i)] for i in range(96)] for s in range(10)]
            
            print(f"\n{'='*60}")
            print(f"Day {global_day+1} (Year {year+1})")
            print(f"{'='*60}")
            
            print(f"\n--- CONTROL ACTIONS ---")
            print(f"  Avg blind position: {np.mean(day_blinds):.1f}")
            print(f"  Avg dimming (uniform): {np.mean([np.mean(d) for d in day_dims]):.3f}")
            
            print(f"\n--- SAFETY (MEAN-BASED) ---")
            in_range_mean = sum(1 for x in day_illum if 500 <= x <= 800)
            below_mean = sum(1 for x in day_illum if x < 500)
            above_mean = sum(1 for x in day_illum if x > 800)
            print(f"  Mean in range [500-800]: {100*in_range_mean/96:.1f}%")
            print(f"  Mean below 500: {100*below_mean/96:.1f}%")
            print(f"  Mean above 800: {100*above_mean/96:.1f}%")
            print(f"  Avg mean illuminance: {np.mean(day_illum):.1f} lux")
            
            print(f"\n--- SAFETY (PER-SENSOR) ---")
            # Check if ANY sensor violates
            sensor_violations = 0
            for t in range(96):
                min_sensor = min(day_sensor_illum[s][t] for s in range(10))
                max_sensor = max(day_sensor_illum[s][t] for s in range(10))
                if min_sensor < 500 or max_sensor > 800:
                    sensor_violations += 1
            print(f"  Per-sensor compliance: {100*(96-sensor_violations)/96:.1f}%")
            print(f"  Min sensor illum: {min(min(day_sensor_illum[s]) for s in range(10)):.1f} lux")
            print(f"  Max sensor illum: {max(max(day_sensor_illum[s]) for s in range(10)):.1f} lux")
            
            print(f"\n--- GLARE ---")
            print(f"  Max DGP: {np.max(day_dgp):.3f}")
            print(f"  Avg DGP: {np.mean(day_dgp):.3f}")
            print(f"  Max vertical illum: {np.max(day_vert):.0f} lux")
            glare_ok = sum(1 for x in day_dgp if x < 0.35)
            print(f"  DGP < 0.35: {100*glare_ok/96:.1f}%")
            
            print(f"\n--- ENERGY ---")
            print(f"  Daily energy: {day_energy:.2f} Wh")
        
        if (global_day + 1) % save_interval == 0:
            results_path = f"{results_dir}/rb_mean_results_day{global_day+1}.npz"
            np.savez(
                results_path,
                rb_actions=np.array(rb_actions_history),
                rb_blinds=np.array(rb_blinds_history),
                illuminances=np.array(timestep_illuminances_avg),
                illuminances_per_sensor=np.array(timestep_illuminances_per_sensor),
                dgp=np.array(dgp_history),
                energies=np.array(timestep_energies),
            )
            print(f"\n  Results saved to {results_path}")
            
            # ================================================================
            # PLOTTING
            # ================================================================
            fig = plt.figure(figsize=(20, 24))
            window = 96
            
            plt.subplot(6, 3, 1)
            rb_avg = [np.mean(a) for a in rb_actions_history]
            plt.plot(rb_avg, alpha=0.7, linewidth=0.5, color='blue')
            plt.title('RB-Mean Dimming (Uniform)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            
            plt.subplot(6, 3, 2)
            plt.plot(rb_blinds_history, alpha=0.7, linewidth=0.5, color='blue')
            plt.title('RB-Mean Blind Position')
            plt.xlabel('Timestep')
            plt.ylabel('Blind Position')
            plt.ylim(-0.5, 16.5)
            plt.grid(True)
            
            plt.subplot(6, 3, 3)
            rb_blind_counts = [rb_blinds_history.count(i) for i in range(17)]
            plt.bar(range(17), rb_blind_counts, alpha=0.7, color='blue')
            plt.title('Blind Action Distribution')
            plt.xlabel('Blind Position')
            plt.ylabel('Count')
            plt.grid(True)
            
            plt.subplot(6, 3, 4)
            plt.plot(timestep_illuminances_avg, linewidth=0.5, color='blue', label='Mean')
            plt.axhline(y=500, color='r', linestyle='--', label='Min (500)')
            plt.axhline(y=800, color='r', linestyle='--', label='Max (800)')
            plt.axhline(y=600, color='g', linestyle=':', label='Target (600)')
            plt.title('Mean Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 5)
            compliance_mean = []
            for i in range(len(timestep_illuminances_avg)):
                start = max(0, i - window + 1)
                in_range = sum(1 for x in timestep_illuminances_avg[start:i+1] if 500 <= x <= 800)
                compliance_mean.append(100 * in_range / (i - start + 1))
            plt.plot(compliance_mean, linewidth=0.5, color='blue')
            plt.title('Mean Illuminance Compliance (Rolling)')
            plt.xlabel('Timestep')
            plt.ylabel('%')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(6, 3, 6)
            # Per-sensor compliance (stricter)
            compliance_sensor = []
            for i in range(len(timestep_illuminances_avg)):
                start = max(0, i - window + 1)
                violations = 0
                for j in range(start, i + 1):
                    min_s = min(timestep_illuminances_per_sensor[s][j] for s in range(10))
                    max_s = max(timestep_illuminances_per_sensor[s][j] for s in range(10))
                    if min_s < 500 or max_s > 800:
                        violations += 1
                compliance_sensor.append(100 * (i - start + 1 - violations) / (i - start + 1))
            plt.plot(compliance_sensor, linewidth=0.5, color='green')
            plt.title('Per-Sensor Compliance (Rolling)')
            plt.xlabel('Timestep')
            plt.ylabel('%')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(6, 3, 7)
            for i in range(10):
                plt.plot(timestep_illuminances_per_sensor[i], label=f'S{i+1}', alpha=0.5, linewidth=0.3)
            plt.axhline(y=500, color='r', linestyle='--')
            plt.axhline(y=800, color='r', linestyle='--')
            plt.title('Per-Sensor Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(6, 3, 8)
            # Sensor spread (max - min)
            sensor_spread = [max(timestep_illuminances_per_sensor[s][i] for s in range(10)) - 
                            min(timestep_illuminances_per_sensor[s][i] for s in range(10)) 
                            for i in range(len(timestep_illuminances_avg))]
            plt.plot(sensor_spread, linewidth=0.5, color='purple')
            plt.title('Sensor Spread (Max - Min)')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.grid(True)
            
            plt.subplot(6, 3, 9)
            plt.hist(sensor_spread, bins=50, alpha=0.7, color='purple')
            plt.title('Sensor Spread Distribution')
            plt.xlabel('Lux')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            plt.subplot(6, 3, 10)
            plt.plot(dgp_history, linewidth=0.5, color='purple')
            plt.axhline(y=0.35, color='r', linestyle='--', label='Imperceptible (0.35)')
            plt.axhline(y=0.40, color='orange', linestyle='--', label='Perceptible (0.40)')
            plt.title('Daylight Glare Probability (DGP)')
            plt.xlabel('Timestep')
            plt.ylabel('DGP')
            plt.ylim(0, 0.6)
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 11)
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
            
            plt.subplot(6, 3, 12)
            plt.plot(vert_illum_history, linewidth=0.5, color='purple')
            plt.axhline(y=2669, color='r', linestyle='--', label='DGP=0.35')
            plt.axhline(y=3473, color='orange', linestyle='--', label='DGP=0.40')
            plt.title('Vertical Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 13)
            plt.plot(timestep_daylight_avg, linewidth=0.5, color='gold')
            plt.title('Daylight Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.grid(True)
            
            plt.subplot(6, 3, 14)
            plt.plot(timestep_energies, linewidth=0.5)
            plt.title('Energy Consumption')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(6, 3, 15)
            rolling_energy = []
            for i in range(len(timestep_energies)):
                start = max(0, i - window + 1)
                rolling_energy.append(np.mean(timestep_energies[start:i+1]))
            plt.plot(rolling_energy, linewidth=0.5, color='orange')
            plt.title('Rolling Avg Energy')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(6, 3, 16)
            plt.hist(timestep_illuminances_avg, bins=50, alpha=0.7, color='blue')
            plt.axvline(x=500, color='r', linestyle='--', label='Min (500)')
            plt.axvline(x=800, color='r', linestyle='--', label='Max (800)')
            plt.axvline(x=600, color='g', linestyle=':', label='Target (600)')
            plt.title('Mean Illuminance Distribution')
            plt.xlabel('Lux')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 17)
            # All sensor values flattened
            all_sensor_values = []
            for s in range(10):
                all_sensor_values.extend(timestep_illuminances_per_sensor[s])
            plt.hist(all_sensor_values, bins=50, alpha=0.7, color='green')
            plt.axvline(x=500, color='r', linestyle='--', label='Min (500)')
            plt.axvline(x=800, color='r', linestyle='--', label='Max (800)')
            plt.title('All Sensor Values Distribution')
            plt.xlabel('Lux')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 18)
            plt.hist(dgp_history, bins=50, alpha=0.7, color='purple')
            plt.axvline(x=0.35, color='r', linestyle='--', label='DGP=0.35')
            plt.axvline(x=0.40, color='orange', linestyle='--', label='DGP=0.40')
            plt.title('DGP Distribution')
            plt.xlabel('DGP')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/plots/detailed_day{global_day+1}.png", dpi=150)
            plt.close()
    
    # ========================================================================
    # END OF YEAR
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"YEAR {year + 1} COMPLETED")
    print(f"{'='*60}")
    
    # Year summary
    year_illum = timestep_illuminances_avg
    year_dgp = dgp_history
    year_energy = timestep_energies
    
    # Mean-based compliance
    in_range_mean = sum(1 for x in year_illum if 500 <= x <= 800)
    
    # Per-sensor compliance (stricter)
    sensor_violations = 0
    for i in range(len(year_illum)):
        min_s = min(timestep_illuminances_per_sensor[s][i] for s in range(10))
        max_s = max(timestep_illuminances_per_sensor[s][i] for s in range(10))
        if min_s < 500 or max_s > 800:
            sensor_violations += 1
    
    glare_ok = sum(1 for x in year_dgp if x < 0.35)
    
    print(f"\n--- YEAR {year+1} SUMMARY ---")
    print(f"  Mean illuminance compliance: {100*in_range_mean/len(year_illum):.1f}%")
    print(f"  Per-sensor compliance: {100*(len(year_illum)-sensor_violations)/len(year_illum):.1f}%")
    print(f"  Glare compliance: {100*glare_ok/len(year_dgp):.1f}%")
    print(f"  Total energy: {sum(year_energy):.0f} Wh")
    print(f"  Avg energy per day: {sum(year_energy)/num_days:.0f} Wh")
    
    # Clear for next year
    rb_actions_history = []
    rb_blinds_history = []
    dgp_history = []
    vert_illum_history = []
    timestep_energies = []
    timestep_illuminances_per_sensor = [[] for _ in range(10)]
    timestep_illuminances_avg = []
    timestep_daylight_avg = []
    
    plt.close('all')
    gc.collect()

print("\n" + "=" * 60)
print("SIMULATION COMPLETED!")
print("=" * 60)
print(f"Total years: {num_years}")
print(f"Total days: {num_years * num_days}")
print(f"\nResults saved to {results_dir}")