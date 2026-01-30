import numpy as np
from utils.sun_pos_day_of_year import get_day_of_year, sunPos
D_lookup = np.load("daylight_lookup.npy")  # (35040, 17, 10)
D_vert_lookup = np.load("Vertical_daylight_lookup1.npy")  # (35040, 17, 10)

L = np.array([
    [401.15861196,  58.4204697,  225.5377067,   49.14087042,  85.19253081,  79.86742317],
    [438.00878779,  48.64657768, 229.29380513,  38.7237036,   61.16027223,  60.86753448],
    [352.96810231, 146.95379596, 186.78037056,  86.44197848,  55.33809343,  52.93346111],
    [ 99.61285736, 396.01190196,  60.83606539, 205.34517382,  50.49735697,  47.5929032 ],
    [ 24.83944566, 428.72332982,  21.09381429, 230.76645939,  41.18817081,  38.40115304],
    [206.82814047,  58.73720228, 415.13020836,  50.31042966,  82.72699835,  79.42208608],
    [211.20090374,  40.79763768, 448.78504733,  45.58753835,  62.2310672,   61.41399961],
    [175.21200584,  78.84503981, 367.71876396, 147.20793241,  53.39780771,  54.50891389],
    [ 60.26461867, 187.53255499, 101.52002782, 393.91797889,  54.3553788,   52.39289599],
    [ 24.48174991, 207.96177056,  24.87482134, 444.56078609,  41.64620997,  39.70489902]
])

# Power vector for energy calculation
POW_VEC = np.array([46.3*8, 46.3*8, 46.3*8, 46.3*8, 57.9*8, 57.9*8])
with open("Wea_files/innsbruck_din5034_clear_t3.5.wea") as f:
    wea_lines = [line.rstrip() for line in f]
    WEA_DATA = [[float(val) for val in line.split(" ")] for line in wea_lines[6:]]


def RADstep_fast(DIM_VEC, blind_id, step, mpc_correction, D_lookup=D_lookup, L=L, D_vert_lookup=D_vert_lookup):
    
    DIM_VEC = np.array(DIM_VEC)
    illum_lights = L @ DIM_VEC
    illum_daylight = D_lookup[step, blind_id, :]
    illum_vertical = D_vert_lookup[step, blind_id, :]
    tot_illum = illum_lights + illum_daylight
    
    Tot_ene_cons_ts = (DIM_VEC @ POW_VEC) * 0.25
    
    timestep_in_day = step % 96
    hour = timestep_in_day / 96.0
    occupied = (8 * 4) <= timestep_in_day < (18 * 4)
    
    simplified_glare_index = (6.22e-5) * illum_vertical + 0.184
    max_dgp = np.max(simplified_glare_index)
    
    if step < len(WEA_DATA):
        wea_step = WEA_DATA[step]
        M = get_day_of_year(wea_step)
        sun_alt, sun_azi = sunPos(wea_step[2], 47, 11.35, 15, M)
    else:
        M = 365
        sun_alt, sun_azi = 0, 180
    
    # ========================================================================
    # REWARD THAT FORCES DAYLIGHT USE
    # ========================================================================
    
    TARGET_MIN, TARGET_MAX = 500, 800
    GLARE_LIMIT = 0.402
    
    if occupied:
        # --- A. COMFORT (same as before) ---
        mean_illum = np.mean(tot_illum)
        if TARGET_MIN <= mean_illum <= TARGET_MAX:
            comfort_reward = 0.2
        else:
            if mean_illum < TARGET_MIN:
                comfort_reward = -((TARGET_MIN - mean_illum) / 300) ** 2
            else:
                comfort_reward = -((mean_illum - TARGET_MAX) / 300) ** 2
        
        # --- B. GLARE ---
        if max_dgp < 0.30:
            glare_reward = 0.1
        elif max_dgp < GLARE_LIMIT:
            glare_reward = 0.0
        else:
            glare_reward = -((max_dgp - GLARE_LIMIT) * 10) ** 2
        
        # --- C. ELECTRIC LIGHT PENALTY (KEY CHANGE) ---
        # Penalize electric light usage PROPORTIONAL to available daylight
        # Check what daylight WOULD be available with fully open blinds
        daylight_if_open = D_lookup[step, 0, :]  # blind_id=0 is fully open
        available_daylight = np.mean(daylight_if_open)
        
        electric_usage = np.mean(illum_lights)
        
        if available_daylight > 300:
            # Plenty of daylight available - HEAVILY penalize electric use
            wasted_potential = electric_usage / (available_daylight + 1)
            electric_penalty = -wasted_potential * 1.5
        elif available_daylight > 100:
            # Some daylight - moderate penalty
            electric_penalty = -(electric_usage / 1000) * 0.5
        else:
            # Low daylight (evening/cloudy) - small penalty
            electric_penalty = -(electric_usage / 1000) * 0.1
        
        # --- D. BLIND POSITION BONUS (KEY CHANGE) ---
        # Direct reward for opening blinds when safe
        if max_dgp < GLARE_LIMIT:
            # No glare issue - reward open blinds
            blind_openness = (16 - blind_id) / 16.0  
            blind_bonus = blind_openness * 0.3
        else:
            # Glare issue - no penalty for closing
            blind_bonus = 0.0
        
        # --- E. DAYLIGHT RATIO BONUS ---
        daylight_used = np.mean(illum_daylight)
        total_light = np.mean(tot_illum) + 1e-6
        daylight_ratio = daylight_used / total_light
        daylight_bonus = daylight_ratio * 0.4  # Strong bonus
        
        # --- F. ENERGY ---
        energy_penalty = -(Tot_ene_cons_ts / 400) * 0.2
        
        # --- G. MPC (minimal) ---
        mpc_penalty = -mpc_correction * .5
        
        reward = (
            comfort_reward +      # [-1, +0.2]
            glare_reward +        # [-big, +0.1]
            electric_penalty +    # [-1.5, 0]  <-- NEW: harsh when daylight available
            blind_bonus +         # [0, +0.3]  <-- NEW: direct blind reward
            daylight_bonus +      # [0, +0.4]  <-- STRONGER
            energy_penalty +      # [-0.2, 0]
            mpc_penalty
        )
        if np.mean(tot_illum) < 300:
            reward -= 0.5  # Harsh penalty for dark rooms when occupied
    else:
        # Unoccupied - same as before
        if Tot_ene_cons_ts < 50.0:
            reward = 0.2
        else:
            reward = -(Tot_ene_cons_ts/ 400)
        reward -= mpc_correction * .5

    done = (step >= 35039)
    
    state = (
        (illum_lights / 1000).tolist() +
        (illum_daylight / 1000).tolist() +
        (illum_vertical / 1000).tolist() +
        [
            math.sin(math.radians(sun_azi)),
            math.cos(math.radians(sun_azi)),
            hour,
            sun_alt/90,
            float(occupied)
        ]
    )
    
    info = {
        "lights_illuminance": illum_lights.tolist(),
        "daylight_illuminance": illum_daylight.tolist(),
        "tot_illuminance": tot_illum,
        "vertical_illuminance": illum_vertical.tolist(),
        "power_consumed": Tot_ene_cons_ts,
        "dgp": max_dgp,
        "blind_id": blind_id,
        "daylight_ratio": daylight_ratio if occupied else 0,
        "available_daylight": available_daylight if occupied else 0,
        "occupied": occupied,
    }
    
    return state, info, reward, done