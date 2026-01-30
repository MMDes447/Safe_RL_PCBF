import os
import gc
from datetime import datetime
import numpy as np
import torch as T
import matplotlib.pyplot as plt

from agents.ppo import PPOAgent1_epch
from mpc_filter import safety_MILP_two_stage
from steps.fast_step import RADstep_fast, D_lookup, D_vert_lookup, L  # Still need for counterfactual reward
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


USE_FAST_STEP = True  # Set False to use full Radiance simulation

def compute_rl_reward(dim_vec, blind_id, timestep, D_lookup, D_vert_lookup, L, mpc_correction=0.0):
    
    TARGET_MIN, TARGET_MAX = 500, 800
    DGP_LIMIT = 0.40
    POW_VEC = np.array([46.3*8, 46.3*8, 46.3*8, 46.3*8, 57.9*8, 57.9*8])
    
    timestep_in_day = timestep % 96
    is_occupied = (8 * 4) <= timestep_in_day < (18 * 4)
    
    dim_vec = np.clip(dim_vec, 0, 1)
    illum_lights = L @ dim_vec
    illum_daylight = D_lookup[timestep, blind_id, :]
    tot_illum = illum_lights + illum_daylight
    
    vert_illum = D_vert_lookup[timestep, blind_id, :]
    dgp = 6.22e-5 * vert_illum + 0.184
    max_dgp = np.max(dgp)
    
    energy = (dim_vec @ POW_VEC) * 0.25
    
    if not is_occupied:
        if energy < 5:
            reward = 0.2
        elif energy == 0:
            reward = 0.6
        else:
            reward = -(energy / 400)
        reward -= mpc_correction * 0.5
        return reward
    
    mean_illum = np.mean(tot_illum)
    min_illum = np.min(tot_illum)
    max_illum = np.max(tot_illum)
    
    if min_illum >= TARGET_MIN and max_illum <= TARGET_MAX:
        comfort = 0.3
    else:
        below_penalty = max(0, TARGET_MIN - min_illum) / 300
        above_penalty = max(0, max_illum - TARGET_MAX) / 300
        comfort = -(below_penalty ** 2 + above_penalty ** 2)
    
    if max_dgp < 0.30:
        glare = 0.1
    elif max_dgp < DGP_LIMIT:
        glare = 0.0
    else:
        glare = -((max_dgp - DGP_LIMIT) * 10) ** 2
    
    energy_penalty = -(energy / 400) * 0.2
    
    daylight_used = np.mean(illum_daylight)
    total_light = mean_illum + 1e-6
    daylight_ratio = daylight_used / total_light
    daylight_bonus = daylight_ratio * 0.2
    
    mpc_penalty = -mpc_correction * 0.5
    
    reward = comfort + glare + energy_penalty + daylight_bonus + mpc_penalty
    return reward



num_days = 365
timesteps_per_day = 96
initial_state = [0] * 10 + [0] * 10 + [0] * 10 + [0.0, 1.0, 0.0, 0.0, 1.0]
log_interval = 1
save_interval = 7
learn_interval = 7
MPC_HORIZON = 2
MPC_DELTA_MAX = 1
MPC_W_BLIND = 0.5

results_dir = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


rl_actions_history = []
mpc_actions_history = []
rl_blinds_history = []
mpc_blinds_history = []
blind_changes_history = []
mpc_corrections = []
mpc_intervened = []
slack_upper_history = []
slack_lower_history = []
slack_rate_history = []
dgp_history = []
vert_illum_history = []
timestep_rewards = []
timestep_energies = []
timestep_illuminances_per_sensor = [[] for _ in range(10)]
timestep_illuminances_avg = []
timestep_daylight_avg = []

prev_u = np.ones(6) * 0.5
prev_x = np.ones(10) * 700


num_years = 20
print(f"Starting MILP MPC training for {num_years} years ({num_years * num_days} days)")
print(f"Step function: {'FAST (lookup)' if USE_FAST_STEP else 'FULL (Radiance)'}")
print(f"MPC Horizon: {MPC_HORIZON}")
print(f"Safety bounds: [500, 1000] lux")
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
            verbose = (global_timestep < 5)
            
            safe_continuous_action, safe_blind_action, diagnostics = safety_MILP_two_stage(
                a_RL=continuous_action,
                b_RL=discrete_action,
                x_prev=prev_x,
                u_prev=prev_u,
                t=lookup_timestep,
                D=D_lookup,
                D_vert=D_vert_lookup,
                N=MPC_HORIZON,
                delta_max=MPC_DELTA_MAX,
                w_blind=MPC_W_BLIND,
                verbose=verbose
            )
            
            # Track actions
            rl_actions_history.append(continuous_action.copy())
            mpc_actions_history.append(safe_continuous_action.copy())
            rl_blinds_history.append(discrete_action)
            mpc_blinds_history.append(safe_blind_action)
            
            # Track diagnostics
            if diagnostics is not None:
                correction_norm = diagnostics['correction_norm']
                blind_changed = diagnostics.get('blind_changed', False)
                mpc_corrections.append(correction_norm)
                mpc_intervened.append(correction_norm > 1e-4 or blind_changed)
                blind_changes_history.append(1 if blind_changed else 0)
                slack_upper_history.append(diagnostics.get('slack_upper', 0))
                slack_lower_history.append(diagnostics.get('slack_lower', 0))
                slack_rate_history.append(diagnostics.get('slack_rate', 0))
            else:
                correction_norm = np.linalg.norm(safe_continuous_action - continuous_action)
                mpc_corrections.append(correction_norm)
                mpc_intervened.append(correction_norm > 1e-4)
                blind_changes_history.append(0)
                slack_upper_history.append(0)
                slack_lower_history.append(0)
                slack_rate_history.append(0)
            
            prev_u = safe_continuous_action.copy()
            
            # ================================================================
            # STEP FUNCTION - CHOOSE FAST OR FULL RADIANCE
            # ================================================================
            if USE_FAST_STEP:
                next_state, info, mpc_reward, done = RADstep_fast(
                    safe_continuous_action,
                    safe_blind_action,
                    lookup_timestep,
                    correction_norm
                )
            else:
                next_state, info, mpc_reward, done = RADstep(
                    DIM_VEC=safe_continuous_action,
                    a_id=safe_blind_action,
                    File_paths=File_paths,
                    res_path=res_path,
                    step=lookup_timestep,
                    mpc_correction=correction_norm
                )
            
            # Counterfactual reward (always uses lookup for speed)
            rl_reward = compute_rl_reward(
                continuous_action,
                discrete_action,
                lookup_timestep,
                D_lookup,
                D_vert_lookup,
                L,
                correction_norm
            )
            
            prev_x = np.array(info["tot_illuminance"])
            
            vert_illum = np.array(info["vertical_illuminance"])
            dgp = 6.22e-5 * vert_illum + 0.184
            dgp_history.append(np.max(dgp))
            vert_illum_history.append(np.max(vert_illum))
            
            agent.remember(
                observation,
                discrete_action,
                continuous_action,
                safe_continuous_action,
                safe_blind_action,
                log_prob,
                val,
                rl_reward,
                done,
                next_state
            )
            
            timestep_rewards.append(rl_reward)
            timestep_energies.append(info["power_consumed"])
            
            tot_illum = info["tot_illuminance"]
            for sensor_idx in range(10):
                timestep_illuminances_per_sensor[sensor_idx].append(tot_illum[sensor_idx])
            timestep_illuminances_avg.append(np.mean(tot_illum))
            timestep_daylight_avg.append(np.mean(info["daylight_illuminance"]))
            
            day_reward += rl_reward
            day_energy += info["power_consumed"]
            observation = next_state
        
        
        
        if (global_day + 1) % learn_interval == 0:
            print(f"\nDay {global_day+1}: Learning from experience...")
            agent.learn()
        
        if (global_day + 1) % log_interval == 0:
            day_corrections = mpc_corrections[-96:]
            day_intervened = mpc_intervened[-96:]
            day_blind_changes = blind_changes_history[-96:]
            day_illum = timestep_illuminances_avg[-96:]
            day_dgp = dgp_history[-96:]
            day_vert = vert_illum_history[-96:]
            
            print(f"\n{'='*60}")
            print(f"Day {global_day+1} (Year {year+1})")
            print(f"{'='*60}")
            
            print(f"\n--- MPC MILP FILTER ---")
            intervention_rate = 100 * sum(day_intervened) / len(day_intervened)
            blind_change_rate = 100 * sum(day_blind_changes) / len(day_blind_changes)
            print(f"  Intervention rate: {intervention_rate:.1f}%")
            print(f"  Blind change rate: {blind_change_rate:.1f}%")
            print(f"  Avg correction: {np.mean(day_corrections):.4f}")
            print(f"  Max correction: {np.max(day_corrections):.4f}")
            
            print(f"\n--- SAFETY ---")
            in_range = sum(1 for x in day_illum if 500 <= x <= 1000)
            below = sum(1 for x in day_illum if x < 500)
            above = sum(1 for x in day_illum if x > 1000)
            print(f"  In range [500-1000]: {100*in_range/96:.1f}%")
            print(f"  Below 500: {100*below/96:.1f}%")
            print(f"  Above 1000: {100*above/96:.1f}%")
            print(f"  Avg illuminance: {np.mean(day_illum):.1f} lux")
            
            print(f"\n--- GLARE (from step) ---")
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
                'mpc_corrections': mpc_corrections,
                'dgp_history': dgp_history,
                'vert_illum_history': vert_illum_history,
            }, checkpoint_path)
            print(f"\n  Checkpoint saved to {checkpoint_path}")
            
            fig = plt.figure(figsize=(24, 32))
            window = 96
            
            plt.subplot(8, 3, 1)
            rl_avg = [np.mean(a) for a in rl_actions_history]
            plt.plot(rl_avg, alpha=0.7, linewidth=0.5)
            plt.title('RL Proposed Dimming (Average)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            
            plt.subplot(8, 3, 2)
            mpc_avg = [np.mean(a) for a in mpc_actions_history]
            plt.plot(mpc_avg, alpha=0.7, linewidth=0.5, color='orange')
            plt.title('MPC Output Dimming (Average)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            
            plt.subplot(8, 3, 3)
            plt.plot(rl_avg, label='RL', alpha=0.6, linewidth=0.5)
            plt.plot(mpc_avg, label='MPC', alpha=0.6, linewidth=0.5)
            plt.title('RL vs MPC Dimming Comparison')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming Level')
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.grid(True)
            
            plt.subplot(8, 3, 4)
            plt.plot(rl_blinds_history, alpha=0.7, linewidth=0.5)
            plt.title('RL Proposed Blind')
            plt.xlabel('Timestep')
            plt.ylabel('Blind Position')
            plt.ylim(-0.5, 16.5)
            plt.grid(True)
            
            plt.subplot(8, 3, 5)
            plt.plot(mpc_blinds_history, alpha=0.7, linewidth=0.5, color='orange')
            plt.title('MPC Output Blind')
            plt.xlabel('Timestep')
            plt.ylabel('Blind Position')
            plt.ylim(-0.5, 16.5)
            plt.grid(True)
            
            plt.subplot(8, 3, 6)
            rolling_blind_change = []
            for i in range(len(blind_changes_history)):
                start = max(0, i - window + 1)
                rate = 100 * sum(blind_changes_history[start:i+1]) / (i - start + 1)
                rolling_blind_change.append(rate)
            plt.plot(rolling_blind_change, linewidth=0.5, color='red')
            plt.title('Blind Change Rate (Rolling)')
            plt.xlabel('Timestep')
            plt.ylabel('Change %')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(8, 3, 7)
            plt.plot(mpc_corrections, linewidth=0.5)
            plt.title('MPC Correction Magnitude')
            plt.xlabel('Timestep')
            plt.ylabel('||MPC - RL||')
            plt.grid(True)
            
            plt.subplot(8, 3, 8)
            rolling_intervention = []
            for i in range(len(mpc_intervened)):
                start = max(0, i - window + 1)
                rate = 100 * sum(mpc_intervened[start:i+1]) / (i - start + 1)
                rolling_intervention.append(rate)
            plt.plot(rolling_intervention, linewidth=0.5)
            plt.title('MPC Intervention Rate (Rolling 1-day)')
            plt.xlabel('Timestep')
            plt.ylabel('Intervention %')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(8, 3, 9)
            plt.hist(mpc_corrections, bins=50)
            plt.title('Correction Distribution')
            plt.xlabel('Correction Magnitude')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            plt.subplot(8, 3, 10)
            plt.plot(timestep_illuminances_avg, linewidth=0.5)
            plt.axhline(y=500, color='r', linestyle='--', label='Min (500)')
            plt.axhline(y=1000, color='r', linestyle='--', label='Max (1000)')
            plt.title('Total Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(8, 3, 11)
            plt.plot(slack_upper_history, label='Upper', alpha=0.7, linewidth=0.5)
            plt.plot(slack_lower_history, label='Lower', alpha=0.7, linewidth=0.5)
            plt.title('Safety Slack Usage')
            plt.xlabel('Timestep')
            plt.ylabel('Slack')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(8, 3, 12)
            plt.plot(slack_rate_history, linewidth=0.5, color='green')
            plt.title('Rate Slack Usage')
            plt.xlabel('Timestep')
            plt.ylabel('Slack')
            plt.grid(True)
            
            plt.subplot(8, 3, 13)
            plt.plot(dgp_history, linewidth=0.5, color='purple')
            plt.axhline(y=0.35, color='r', linestyle='--', label='Imperceptible (0.35)')
            plt.axhline(y=0.40, color='orange', linestyle='--', label='Perceptible (0.40)')
            plt.title('Daylight Glare Probability (DGP) - from Step')
            plt.xlabel('Timestep')
            plt.ylabel('DGP')
            plt.ylim(0, 0.6)
            plt.legend()
            plt.grid(True)
            
            plt.subplot(8, 3, 14)
            plt.plot(vert_illum_history, linewidth=0.5, color='purple')
            plt.axhline(y=2669, color='r', linestyle='--', label='DGP=0.35')
            plt.axhline(y=3473, color='orange', linestyle='--', label='DGP=0.40')
            plt.title('Vertical Illuminance (from Step)')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(8, 3, 15)
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
            
            plt.subplot(8, 3, 16)
            for i in range(10):
                plt.plot(timestep_illuminances_per_sensor[i], label=f'S{i+1}', alpha=0.5, linewidth=0.3)
            plt.axhline(y=500, color='r', linestyle='--')
            plt.axhline(y=1000, color='r', linestyle='--')
            plt.title('Per-Sensor Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(8, 3, 17)
            compliance = []
            for i in range(len(timestep_illuminances_avg)):
                start = max(0, i - window + 1)
                in_range = sum(1 for x in timestep_illuminances_avg[start:i+1] if 500 <= x <= 1000)
                compliance.append(100 * in_range / (i - start + 1))
            plt.plot(compliance, linewidth=0.5, color='green')
            plt.title('Safety Compliance (Rolling)')
            plt.xlabel('Timestep')
            plt.ylabel('%')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(8, 3, 18)
            plt.plot(timestep_daylight_avg, linewidth=0.5, color='gold')
            plt.title('Daylight Illuminance')
            plt.xlabel('Timestep')
            plt.ylabel('Lux')
            plt.grid(True)
            
            plt.subplot(8, 3, 19)
            plt.plot(timestep_energies, linewidth=0.5)
            plt.title('Energy Consumption')
            plt.xlabel('Timestep')
            plt.ylabel('Wh')
            plt.grid(True)
            
            plt.subplot(8, 3, 20)
            plt.plot(timestep_rewards, linewidth=0.5)
            plt.title('Reward')
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.grid(True)
            
            plt.subplot(8, 3, 21)
            rl_blind_counts = [rl_blinds_history.count(i) for i in range(17)]
            mpc_blind_counts = [mpc_blinds_history.count(i) for i in range(17)]
            x = np.arange(17)
            width = 0.35
            plt.bar(x - width/2, rl_blind_counts, width, label='RL', alpha=0.7)
            plt.bar(x + width/2, mpc_blind_counts, width, label='MPC', alpha=0.7)
            plt.title('Blind Action Distribution')
            plt.xlabel('Blind Position')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(8, 3, 22)
            for i in range(6):
                rl_light = [a[i] for a in rl_actions_history]
                plt.plot(rl_light, label=f'L{i+1}', alpha=0.5, linewidth=0.3)
            plt.title('RL Proposed (Per Light)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming')
            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(8, 3, 23)
            for i in range(6):
                mpc_light = [a[i] for a in mpc_actions_history]
                plt.plot(mpc_light, label=f'L{i+1}', alpha=0.5, linewidth=0.3)
            plt.title('MPC Output (Per Light)')
            plt.xlabel('Timestep')
            plt.ylabel('Dimming')
            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=6)
            plt.grid(True)
            
            plt.subplot(8, 3, 24)
            for i in range(6):
                corr_i = [abs(mpc_actions_history[t][i] - rl_actions_history[t][i]) 
                          for t in range(len(rl_actions_history))]
                plt.plot(corr_i, label=f'L{i+1}', alpha=0.5, linewidth=0.3)
            plt.title('Per-Light Correction')
            plt.xlabel('Timestep')
            plt.ylabel('|MPC - RL|')
            plt.legend(fontsize=6)
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
    
    # Clear tracking lists to prevent memory overflow
    rl_actions_history = []
    mpc_actions_history = []
    rl_blinds_history = []
    mpc_blinds_history = []
    blind_changes_history = []
    mpc_corrections = []
    mpc_intervened = []
    slack_upper_history = []
    slack_lower_history = []
    slack_rate_history = []
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