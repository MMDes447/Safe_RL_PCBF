import numpy as np
import math
from utils.sun_pos_day_of_year import get_day_of_year, sunPos
import os
def RADstep(DIM_VEC, a_id, File_paths, res_path, step, mpc_correction):
   
    
    # ========================================================================
    # DIMMER CONTROL
    # ========================================================================
    st1 = "4 " + str(DIM_VEC[0]) + " 0.12 1.5 0.06\n"
    st2 = "4 " + str(DIM_VEC[1]) + " 0.12 1.5 0.06\n"
    st3 = "4 " + str(DIM_VEC[2]) + " 0.12 1.5 0.06\n"
    st4 = "4 " + str(DIM_VEC[3]) + " 0.12 1.5 0.06\n"
    st5 = "1 " + str(DIM_VEC[4] * 11.4377) + "\n"
    st6 = "1 " + str(DIM_VEC[5] * 11.4377) + "\n"
    Dims = [st1, st2, st3, st4, st5, st6]
    
    for i, (path, line_idx) in enumerate(File_paths):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines[line_idx] = Dims[i]
        with open(path, "w") as f:
            f.writelines(lines)

    # ========================================================================
    # ELECTRIC LIGHTS SIMULATION (rtrace)
    # ========================================================================
    os.system(f"rtrace -n 8 -I+ -ab 2 -ad 256 -as 64 -aa 0.35 -ar 32 -dc 0.7 -dt 0.5 -ds 0.2 -dr 1 -lr 4 -lw 5e-3 -h scene654.oct < lines.pts > {res_path}")
    
    with open(res_path, "r") as res:
        illum_list = [line.rstrip() for line in res]
        illum_arr = np.array([[float(val) for val in i.split("\t")] for i in illum_list])
        illum_arr_lights = np.array([
            179 * (illum_arr[i][0] * 0.265 + illum_arr[i][1] * 0.670 + 0.065 * illum_arr[i][2])
            for i in range(len(illum_arr))
        ])

    # ========================================================================
    # SKY VECTOR GENERATION
    # ========================================================================
    with open("Wea_files/innsbruck_din5034_clear_t3.5.wea") as weat:
        weat_lines = [line.rstrip() for line in weat]
        weat_data = [[float(val) for val in i.split(" ")] for i in weat_lines[6:]]
    
    wea_step = weat_data[step]
    sky_cmd = f"gendaylit {wea_step[0]} {wea_step[1]} {wea_step[2]} -m -15 -o -11.35 -a 47.27 -W {wea_step[3]} {wea_step[4]} | genskyvec -m 4 -c 1 1 1 > Sky_on_the_fly/sky.vec"
    os.system(sky_cmd)
    sky_vec = "Sky_on_the_fly/sky.vec"

    # ========================================================================
    # BLIND STATE MAPPING
    # ========================================================================
    TILTS = ["ven0", "ven30", "ven60", "ven90"]
    CLEAR = "clear"
    
    if not (0 <= a_id <= 16):
        raise ValueError("a_id must be in [0, 16]")
    
    if a_id == 0:
        acts = [CLEAR, CLEAR, CLEAR, CLEAR]
    else:
        lift = 1 + (a_id - 1) // 4
        tilt = TILTS[(a_id - 1) % 4]
        acts = [tilt] * lift + [CLEAR] * (4 - lift)

    # ========================================================================
    # WORKPLANE DAYLIGHT (3-phase method)
    # ========================================================================
    os.system(f"dctimestep V_mats/V1.vmx bl_state/{acts[0]}.xml D_mats/D1.dmx {sky_vec} > res/il_reads1.dat")
    os.system(f"dctimestep V_mats/V2.vmx bl_state/{acts[1]}.xml D_mats/D2.dmx {sky_vec} > res/il_reads2.dat")
    os.system(f"dctimestep V_mats/V3.vmx bl_state/{acts[2]}.xml D_mats/D3.dmx {sky_vec} > res/il_reads3.dat")
    os.system(f"dctimestep V_mats/V4.vmx bl_state/{acts[3]}.xml D_mats/D4.dmx {sky_vec} > res/il_reads4.dat")

    ill_day = np.zeros(10)
    for fname in ["res/il_reads1.dat", "res/il_reads2.dat", "res/il_reads3.dat", "res/il_reads4.dat"]:
        with open(fname) as f:
            lines = [line.rstrip() for line in f]
            arr = np.array([[float(val) for val in line.split(" ")] for line in lines[9:]])
            ill_day += np.array([179 * (arr[i][0] * 0.265 + arr[i][1] * 0.670 + 0.065 * arr[i][2]) for i in range(len(arr))])

    # ========================================================================
    # VERTICAL ILLUMINANCE (3-phase method with V_vert matrices)
    # ========================================================================
    os.system(f"dctimestep V_mats/V_vert1.vmx bl_state/{acts[0]}.xml D_mats/D1.dmx {sky_vec} > res/il_vert1.dat")
    os.system(f"dctimestep V_mats/V_vert2.vmx bl_state/{acts[1]}.xml D_mats/D2.dmx {sky_vec} > res/il_vert2.dat")
    os.system(f"dctimestep V_mats/V_vert3.vmx bl_state/{acts[2]}.xml D_mats/D3.dmx {sky_vec} > res/il_vert3.dat")
    os.system(f"dctimestep V_mats/V_vert4.vmx bl_state/{acts[3]}.xml D_mats/D4.dmx {sky_vec} > res/il_vert4.dat")

    ill_vert = np.zeros(10)
    for fname in ["res/il_vert1.dat", "res/il_vert2.dat", "res/il_vert3.dat", "res/il_vert4.dat"]:
        with open(fname) as f:
            lines = [line.rstrip() for line in f]
            arr = np.array([[float(val) for val in line.split(" ")] for line in lines[9:]])
            ill_vert += np.array([179 * (arr[i][0] * 0.265 + arr[i][1] * 0.670 + 0.065 * arr[i][2]) for i in range(len(arr))])

    # ========================================================================
    # SOLAR POSITION
    # ========================================================================
    M = get_day_of_year(wea_step)
    sun_alt, sun_azi = sunPos(wea_step[2], 47, 11.35, 15, M)

    # ========================================================================
    # DERIVED QUANTITIES
    # ========================================================================
    tot_illum = ill_day + illum_arr_lights
    timestep_in_day = step % 96
    hour = timestep_in_day / 96.0
    occupied = (8 * 4) <= timestep_in_day < (18 * 4)
    done = (step >= 35039)
    
    DIM_VEC_arr = np.array(DIM_VEC)
    POW_VEC = np.array([46.3*8, 46.3*8, 46.3*8, 46.3*8, 57.9*8, 57.9*8])
    Tot_ene_cons_ts = DIM_VEC_arr @ POW_VEC * 0.25

    # Simplified DGP
    simplified_glare_index = (6.22e-5) * ill_vert + 0.184
    max_dgp = np.max(simplified_glare_index)

    # ========================================================================
    # REWARD
    # ========================================================================
    TARGET_MIN, TARGET_MAX = 500, 800
    GLARE_LIMIT = 0.40

    if occupied:
        mean_illum = np.mean(tot_illum)
        min_illum = np.min(tot_illum)
        max_illum = np.max(tot_illum)

        # Comfort
        if min_illum >= TARGET_MIN and max_illum <= TARGET_MAX:
            comfort = 0.3
        else:
            below = max(0, TARGET_MIN - min_illum) / 300
            above = max(0, max_illum - TARGET_MAX) / 300
            comfort = -(below**2 + above**2)

        # Glare
        if max_dgp < 0.30:
            glare = 0.1
        elif max_dgp < GLARE_LIMIT:
            glare = 0.0
        else:
            glare = -((max_dgp - GLARE_LIMIT) * 10) ** 2

        # Energy
        energy_penalty = -(Tot_ene_cons_ts / 400) * 0.2

        # Daylight bonus
        daylight_ratio = np.mean(ill_day) / (mean_illum + 1e-6)
        daylight_bonus = daylight_ratio * 0.2

        # MPC penalty
        mpc_penalty = -mpc_correction * 0.5

        reward = comfort + glare + energy_penalty + daylight_bonus + mpc_penalty
    else:
        if Tot_ene_cons_ts < 5:
            reward = 0.2
        else:
            reward = -(Tot_ene_cons_ts / 400)
        reward -= mpc_correction * 0.5
        daylight_ratio = 0

    # ========================================================================
    # STATE (35-dim)
    # ========================================================================
    state = (
        (illum_arr_lights / 1000).tolist() +      # 10
        (ill_day / 1000).tolist() +               # 10
        (ill_vert / 1000).tolist() +              # 10
        [
            math.sin(math.radians(sun_azi)),      # 1
            math.cos(math.radians(sun_azi)),      # 1
            hour,                                  # 1
            sun_alt / 90,                         # 1
            float(occupied)                        # 1
        ]
    )  # Total: 35

    # ========================================================================
    # INFO
    # ========================================================================
    info = {
        "lights_illuminance": illum_arr_lights.tolist(),
        "daylight_illuminance": ill_day.tolist(),
        "tot_illuminance": tot_illum,
        "vertical_illuminance": ill_vert.tolist(),
        "power_consumed": Tot_ene_cons_ts,
        "dgp": max_dgp,
        "blind_id": a_id,
        "daylight_ratio": daylight_ratio,
        "occupied": occupied,
        "day_of_year": M,
    }

    return state, info, reward, done