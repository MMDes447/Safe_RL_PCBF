import os
import gc
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from steps.fast_step import RADstep_fast, D_lookup, D_vert_lookup, L
from steps.RAD_STEP import RADstep


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


def rule_based_controller_per_sensor(timestep, D_lookup, D_vert_lookup, L):
    """Per-sensor rule-based controller with balanced blind selection and zonal dimming."""
    TARGET_MIN = 500
    TARGET_MAX = 800
    DGP_LIMIT = 0.35
    
    timestep_in_day = timestep % 96
    is_occupied = (8 * 4) <= timestep_in_day < (18 * 4)
    
    if not is_occupied:
        return np.zeros(6), 0
    
 
    best_blind = 0
    best_score = -np.inf
    
    for blind_id in range(17):
        daylight = D_lookup[timestep, blind_id, :]
        vert_illum = D_vert_lookup[timestep, blind_id, :]
        dgp = 6.22e-5 * vert_illum + 0.184
        max_dgp = np.max(dgp)
        
        if max_dgp >= DGP_LIMIT:
            continue
        
        min_day = np.min(daylight)
        max_day = np.max(daylight)
        
        below_penalty = max(0, 300 - min_day)
        above_penalty = max(0, max_day - 750)
        
        score = min_day - 2 * above_penalty - below_penalty
        
        if score > best_score:
            best_score = score
            best_blind = blind_id
    
    # Fallback if no config meets glare limit
    if best_score == -np.inf:
        for blind_id in range(16, -1, -1):
            vert_illum = D_vert_lookup[timestep, blind_id, :]
            dgp = 6.22e-5 * vert_illum + 0.184
            max_dgp = np.max(dgp)
            if max_dgp < 0.45:
                best_blind = blind_id
                break
        else:
            best_blind = 16
    
    
    daylight = D_lookup[timestep, best_blind, :]
    
    dim_level = np.zeros(6)
    max_iterations = 20
    
    for _ in range(max_iterations):
        illum_electric = L @ dim_level
        tot_illum = daylight + illum_electric
        
        min_illum = np.min(tot_illum)
        max_illum = np.max(tot_illum)
        
        if min_illum >= TARGET_MIN and max_illum <= TARGET_MAX:
            break
        
        if min_illum < TARGET_MIN:
            deficit_per_sensor = np.maximum(0, TARGET_MIN - tot_illum)
            
            for zone in range(6):
                zone_contribution = L[:, zone]
                weighted_deficit = np.sum(deficit_per_sensor * zone_contribution) / (np.sum(zone_contribution) + 1e-6)
                dim_level[zone] = np.clip(dim_level[zone] + weighted_deficit / np.mean(zone_contribution) * 0.1, 0, 1)
        
        if max_illum > TARGET_MAX:
            excess_per_sensor = np.maximum(0, tot_illum - TARGET_MAX)
            
            for zone in range(6):
                zone_contribution = L[:, zone]
                weighted_excess = np.sum(excess_per_sensor * zone_contribution) / (np.sum(zone_contribution) + 1e-6)
                dim_level[zone] = np.clip(dim_level[zone] - weighted_excess / np.mean(zone_contribution) * 0.1, 0, 1)
    
    return dim_level, best_blind


num_days = 365
timesteps_per_day = 96
log_interval = 1
save_interval = 7

results_dir = f"results_RB_sensor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/plots", exist_ok=True)


rb_actions_history = []
rb_blinds_history = []
dgp_history = []
vert_illum_history = []
timestep_energies = []
timestep_illuminances_per_sensor = [[] for _ in range(10)]
timestep_illuminances_avg = []
timestep_daylight_avg = []


num_years = 1  # Rule-based doesn't learn, 1 year is enough
print(f"Starting Rule-Based (per-sensor) simulation for {num_years} years ({num_years * num_days} days)")
print(f"Step function: {'FAST (lookup)' if USE_FAST_STEP else 'FULL (Radiance)'}")
print(f"Safety bounds: [500, 800] lux")
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
            
            
            rb_continuous, rb_blind = rule_based_controller_per_sensor(
                lookup_timestep, D_lookup, D_vert_lookup, L
            )
            
            # Track actions
            rb_actions_history.append(rb_continuous.copy())
            rb_blinds_history.append(rb_blind)
            
            
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
            
            print(f"\n{'='*60}")
            print(f"Day {global_day+1} (Year {year+1})")
            print(f"{'='*60}")
            
            print(f"\n--- CONTROL ACTIONS ---")
            print(f"  Avg blind position: {np.mean(day_blinds):.1f}")
            print(f"  Avg dimming: {np.mean([np.mean(d) for d in day_dims]):.3f}")
            
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
            
            print(f"\n--- ENERGY ---")
            print(f"  Daily energy: {day_energy:.2f} Wh")
        
        if (global_day + 1) % save_interval == 0:
            results_path = f"{results_dir}/rb_results_day{global_day+1}.npz"
            np.savez(
                results_path,
                rb_actions=np.array(rb_actions_history),
                rb_blinds=np.array(rb_blinds_history),
                illuminances=np.array(timestep_illuminances_avg),
                dgp=np.array(dgp_history),
                energies=np.array(timestep_energies),
            )
            print(f"\n  Results saved to {results_path}")
            
            
            fig = plt.figure(figsize=(20, 20))
            window = 96
            
            plt.subplot(5, 3, 1)
            rb_avg = [np.mean(a) for a in rb_actions_history]
            plt.plot(rb_avg, alpha=0.7, linewidth=0.5, color='green')
            plt.title('RB Dimming (Average)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            
            plt.subplot(5, 3, 2)
            plt.plot(rb_blinds_history, alpha=0.7, linewidth=0.5, color='green')
            plt.title('RB Blind Position')
            plt.xlabel('Timestep')
            plt.ylabel('Blind Position')
            plt.ylim(-0.5, 16.5)
            plt.grid(True)
            
            plt.subplot(5, 3, 3)
            rb_blind_counts = [rb_blinds_history.count(i) for i in range(17)]
            plt.bar(range(17), rb_blind_counts, alpha=0.7, color='green')
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
            plt.plot(timestep_daylight_avg, linewidth=0.5, color='gold')
            plt.title('Daylight Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.grid(True)
            
            plt.subplot(5, 3, 11)
            plt.plot(timestep_energies, linewidth=0.5)
            plt.title('Energy Consumption')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(5, 3, 12)
            rolling_energy = []
            for i in range(len(timestep_energies)):
                start = max(0, i - window + 1)
                rolling_energy.append(np.mean(timestep_energies[start:i+1]))
            plt.plot(rolling_energy, linewidth=0.5, color='orange')
            plt.title('Rolling Avg Energy')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(5, 3, 13)
            for i in range(6):
                rb_light = [a[i] for a in rb_actions_history]
                plt.plot(rb_light, label=f'L{i+1}', alpha=0.5, linewidth=0.3)
            plt.title('Per-Light Dimming')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming')
            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(5, 3, 14)
            plt.hist(timestep_illuminances_avg, bins=50, alpha=0.7)
            plt.axvline(x=500, color='r', linestyle='--', label='Min (500)')
            plt.axvline(x=800, color='r', linestyle='--', label='Max (800)')
            plt.title('Illuminance Distribution')
            plt.xlabel('Lux')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(5, 3, 15)
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