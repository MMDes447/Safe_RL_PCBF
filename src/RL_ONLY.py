import os
import gc
from datetime import datetime
import numpy as np
import torch as T
import matplotlib.pyplot as plt

from agents.ppo import PPOAgent1_epch
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
USE_FAST_STEP = True  # Set False to use full Radiance simulation

# ============================================================================
# TRAINING CONFIG
# ============================================================================
num_days = 365
timesteps_per_day = 96
initial_state = [0] * 10 + [0] * 10 + [0] * 10 + [0.0, 1.0, 0.0, 0.0, 1.0]
log_interval = 1
save_interval = 7
learn_interval = 7

results_dir = f"training_results_RL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/plots", exist_ok=True)

agent = PPOAgent1_epch(
    n_actions=17,
    input_dims=35,
    continuous_dim=6,
    alpha=0.0003,
    batch_size=96,
    n_epochs=10,
    gae_lambda=0.95,
    gamma=0.99,
    policy_clip=0.2
)

resume_checkpoint = ""
start_day = 0
start_year = 0

if resume_checkpoint and os.path.exists(resume_checkpoint):
    print(f"Loading checkpoint: {resume_checkpoint}")
    checkpoint = T.load(resume_checkpoint)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"Resuming from day {start_day} (year {start_year + 1})")

# ============================================================================
# TRACKING VARIABLES
# ============================================================================
rl_actions_history = []
rl_blinds_history = []
dgp_history = []
vert_illum_history = []
timestep_rewards = []
timestep_energies = []
timestep_illuminances_per_sensor = [[] for _ in range(10)]
timestep_illuminances_avg = []
timestep_daylight_avg = []

# ============================================================================
# TRAINING LOOP
# ============================================================================
num_years = 20
print(f"Starting RL training for {num_years} years ({num_years * num_days} days)")
print(f"Step function: {'FAST (lookup)' if USE_FAST_STEP else 'FULL (Radiance)'}")
print(f"Safety bounds: [500, 800] lux")
print("=" * 60)

for year in range(start_year, num_years):
    day_start = start_day % num_days if year == start_year else 0
    
    print(f"\n{'#'*60}")
    print(f"# YEAR {year + 1}/{num_years}")
    print(f"{'#'*60}")
    
    for day in range(day_start, num_days):
        global_day = year * num_days + day
        observation = initial_state.copy()
        day_reward = 0
        day_energy = 0
        start_timestep = global_day * timesteps_per_day
        
        for timestep_in_day in range(timesteps_per_day):
            global_timestep = start_timestep + timestep_in_day
            lookup_timestep = global_timestep % 35040
            
            discrete_action, continuous_action, log_prob, val = agent.choose_action(observation)
            
            # Track actions
            rl_actions_history.append(continuous_action.copy())
            rl_blinds_history.append(discrete_action)
            
            # ================================================================
            # STEP FUNCTION - DIRECT RL ACTION (NO MPC FILTER)
            # ================================================================
            if USE_FAST_STEP:
                next_state, info, reward, done = RADstep_fast(
                    continuous_action,
                    discrete_action,
                    lookup_timestep,
                    mpc_correction=0.0  # No MPC correction
                )
            else:
                next_state, info, reward, done = RADstep(
                    DIM_VEC=continuous_action,
                    a_id=discrete_action,
                    File_paths=File_paths,
                    res_path=res_path,
                    step=lookup_timestep,
                    mpc_correction=0.0  # No MPC correction
                )
            
            vert_illum = np.array(info["vertical_illuminance"])
            dgp = 6.22e-5 * vert_illum + 0.184
            dgp_history.append(np.max(dgp))
            vert_illum_history.append(np.max(vert_illum))
            
            agent.remember(
                observation,
                discrete_action,
                continuous_action,
                continuous_action,   # No MPC, same as RL
                discrete_action,     # No MPC, same as RL
                log_prob,
                val,
                reward,
                done,
                next_state
            )
            
            timestep_rewards.append(reward)
            timestep_energies.append(info["power_consumed"])
            
            tot_illum = info["tot_illuminance"]
            for sensor_idx in range(10):
                timestep_illuminances_per_sensor[sensor_idx].append(tot_illum[sensor_idx])
            timestep_illuminances_avg.append(np.mean(tot_illum))
            timestep_daylight_avg.append(np.mean(info["daylight_illuminance"]))
            
            day_reward += reward
            day_energy += info["power_consumed"]
            observation = next_state
        
        if (global_day + 1) % learn_interval == 0:
            print(f"\nDay {global_day+1}: Learning from experience...")
            agent.learn()
        
        if (global_day + 1) % log_interval == 0:
            day_illum = timestep_illuminances_avg[-96:]
            day_dgp = dgp_history[-96:]
            day_vert = vert_illum_history[-96:]
            
            print(f"\n{'='*60}")
            print(f"Day {global_day+1} (Year {year+1})")
            print(f"{'='*60}")
            
            print(f"\n--- SAFETY ---")
            in_range = sum(1 for x in day_illum if 500 <= x <= 800)
            below = sum(1 for x in day_illum if x < 500)
            above = sum(1 for x in day_illum if x > 800)
            print(f"  In range [500-800]: {100*in_range/96:.1f}%")
            print(f"  Below 500: {100*below/96:.1f}%")
            print(f"  Above 800: {100*above/96:.1f}%")
            print(f"  Avg illuminance: {np.mean(day_illum):.1f} lux")
            
            print(f"\n--- GLARE ---")
            print(f"  Max DGP: {np.max(day_dgp):.3f}")
            print(f"  Avg DGP: {np.mean(day_dgp):.3f}")
            print(f"  Max vertical illum: {np.max(day_vert):.0f} lux")
            glare_ok = sum(1 for x in day_dgp if x < 0.35)
            print(f"  DGP < 0.35: {100*glare_ok/96:.1f}%")
            
            print(f"\n--- PERFORMANCE ---")
            print(f"  Daily reward: {day_reward:.4f}")
            print(f"  Daily energy: {day_energy:.2f} Wh")
        
        if (global_day + 1) % save_interval == 0:
            checkpoint_path = f"{results_dir}/agent_checkpoint_day{global_day+1}.pt"
            T.save({
                'day': global_day,
                'year': year,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'timestep_rewards': timestep_rewards,
                'dgp_history': dgp_history,
                'vert_illum_history': vert_illum_history,
            }, checkpoint_path)
            print(f"\n  Checkpoint saved to {checkpoint_path}")
            
            # ================================================================
            # PLOTTING (SIMPLIFIED - NO MPC)
            # ================================================================
            fig = plt.figure(figsize=(20, 24))
            window = 96
            
            plt.subplot(6, 3, 1)
            rl_avg = [np.mean(a) for a in rl_actions_history]
            plt.plot(rl_avg, alpha=0.7, linewidth=0.5)
            plt.title('RL Dimming (Average)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            
            plt.subplot(6, 3, 2)
            plt.plot(rl_blinds_history, alpha=0.7, linewidth=0.5)
            plt.title('RL Blind Position')
            plt.xlabel('Timestep')
            plt.ylabel('Blind Position')
            plt.ylim(-0.5, 16.5)
            plt.grid(True)
            
            plt.subplot(6, 3, 3)
            rl_blind_counts = [rl_blinds_history.count(i) for i in range(17)]
            plt.bar(range(17), rl_blind_counts, alpha=0.7)
            plt.title('Blind Action Distribution')
            plt.xlabel('Blind Position')
            plt.ylabel('Count')
            plt.grid(True)
            
            plt.subplot(6, 3, 4)
            plt.plot(timestep_illuminances_avg, linewidth=0.5)
            plt.axhline(y=500, color='r', linestyle='--', label='Min (500)')
            plt.axhline(y=800, color='r', linestyle='--', label='Max (800)')
            plt.title('Total Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 5)
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
            
            plt.subplot(6, 3, 6)
            for i in range(10):
                plt.plot(timestep_illuminances_per_sensor[i], label=f'S{i+1}', alpha=0.5, linewidth=0.3)
            plt.axhline(y=500, color='r', linestyle='--')
            plt.axhline(y=800, color='r', linestyle='--')
            plt.title('Per-Sensor Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(6, 3, 7)
            plt.plot(dgp_history, linewidth=0.5, color='purple')
            plt.axhline(y=0.35, color='r', linestyle='--', label='Imperceptible (0.35)')
            plt.axhline(y=0.40, color='orange', linestyle='--', label='Perceptible (0.40)')
            plt.title('Daylight Glare Probability (DGP)')
            plt.xlabel('Timestep')
            plt.ylabel('DGP')
            plt.ylim(0, 0.6)
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 8)
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
            
            plt.subplot(6, 3, 9)
            plt.plot(vert_illum_history, linewidth=0.5, color='purple')
            plt.axhline(y=2669, color='r', linestyle='--', label='DGP=0.35')
            plt.axhline(y=3473, color='orange', linestyle='--', label='DGP=0.40')
            plt.title('Vertical Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(6, 3, 10)
            plt.plot(timestep_daylight_avg, linewidth=0.5, color='gold')
            plt.title('Daylight Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.grid(True)
            
            plt.subplot(6, 3, 11)
            plt.plot(timestep_energies, linewidth=0.5)
            plt.title('Energy Consumption')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(6, 3, 12)
            plt.plot(timestep_rewards, linewidth=0.5)
            plt.title('Reward')
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.grid(True)
            
            plt.subplot(6, 3, 13)
            for i in range(6):
                rl_light = [a[i] for a in rl_actions_history]
                plt.plot(rl_light, label=f'L{i+1}', alpha=0.5, linewidth=0.3)
            plt.title('Per-Light Dimming')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming')
            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(6, 3, 14)
            rolling_reward = []
            for i in range(len(timestep_rewards)):
                start = max(0, i - window + 1)
                rolling_reward.append(np.mean(timestep_rewards[start:i+1]))
            plt.plot(rolling_reward, linewidth=0.5, color='blue')
            plt.title('Rolling Avg Reward')
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
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
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/plots/detailed_day{global_day+1}.png", dpi=150)
            plt.close()
    
    # ========================================================================
    # END OF YEAR: FLUSH PLOTS AND CLEAR TRACKING LISTS
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"YEAR {year + 1} COMPLETED - Flushing plot memory")
    print(f"{'='*60}")
    
    rl_actions_history = []
    rl_blinds_history = []
    dgp_history = []
    vert_illum_history = []
    timestep_rewards = []
    timestep_energies = []
    timestep_illuminances_per_sensor = [[] for _ in range(10)]
    timestep_illuminances_avg = []
    timestep_daylight_avg = []
    
    plt.close('all')
    gc.collect()

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
print(f"Total years: {num_years}")
print(f"Total days: {num_years * num_days}")
print(f"\nResults saved to {results_dir}")