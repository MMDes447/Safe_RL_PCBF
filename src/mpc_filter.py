import numpy as np
import cvxpy as cp
import torch as T
import matplotlib.pyplot as plt
from datetime import datetime
import os
import gc

D_lookup = np.load("daylight_lookup.npy")
D_vert_lookup = np.load("Vertical_daylight_lookup1.npy") 

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


def safety_MILP_two_stage(a_RL, b_RL, x_prev, u_prev, t, D, D_vert,
                        N=2, delta_max=1, w_blind=0.5, verbose=False):
    
    n_blinds = 17
    n_sensors = 10
    n_lights = 6
    
    PATH_MIN, PATH_MAX = 500, 800
    TERM_MIN, TERM_MAX = 550, 750
    UNOCC_MIN, UNOCC_MAX = 0, 50
    GLARE_MAX = 3500 

    if t + N >= D.shape[0]:
        return np.clip(a_RL, 0, 1), b_RL, None
    
    def is_occupied(timestep):
        timestep_in_day = timestep % 96
        return (8 * 4) <= timestep_in_day < (18 * 4)
    
    occupancy = [is_occupied(t + k + 1) for k in range(N)]
    
    u1 = cp.Variable((N, n_lights))
    delta1 = cp.Variable((N, n_blinds), boolean=True)
    x1 = cp.Variable((N + 1, n_sensors))
    x1_vert = cp.Variable((N + 1, n_sensors)) 
    xi_upper = cp.Variable((N, n_sensors), nonneg=True)
    xi_lower = cp.Variable((N, n_sensors), nonneg=True)
    xi_rate = cp.Variable((N, n_lights), nonneg=True)
    xi_glare = cp.Variable((N, n_sensors), nonneg=True)

    constraints_1 = []
    constraints_1.append(x1[0] == x_prev)
    
    for k in range(N):
        daylight_k = sum(delta1[k, m] * D[t + k + 1, m, :] for m in range(n_blinds))
        constraints_1.append(x1[k + 1] == L @ u1[k] + daylight_k)
        daylight_vert_k = sum(delta1[k, m] * D_vert[t + k + 1, m, :] for m in range(n_blinds))
        constraints_1.append(x1_vert[k + 1] == daylight_vert_k)
        constraints_1.append(cp.sum(delta1[k, :]) == 1)
        constraints_1.append(u1[k] >= 0)
        constraints_1.append(u1[k] <= 1)
        
        if k == 0:
            constraints_1.append(u1[k] - u_prev <= delta_max + xi_rate[k])
            constraints_1.append(u1[k] - u_prev >= -delta_max - xi_rate[k])
        else:
            constraints_1.append(u1[k] - u1[k-1] <= delta_max + xi_rate[k])
            constraints_1.append(u1[k] - u1[k-1] >= -delta_max - xi_rate[k])
        
        if occupancy[k]:
            if k < N - 1:
                constraints_1.append(x1[k + 1] <= PATH_MAX + xi_upper[k])
                constraints_1.append(x1[k + 1] >= PATH_MIN - xi_lower[k])
            else:
                constraints_1.append(x1[k + 1] <= TERM_MAX + xi_upper[k])
                constraints_1.append(x1[k + 1] >= TERM_MIN - xi_lower[k])
            constraints_1.append(x1_vert[k + 1] <= GLARE_MAX + xi_glare[k])
        else:
            constraints_1.append(x1[k + 1] <= UNOCC_MAX + xi_upper[k])
            constraints_1.append(x1[k + 1] >= UNOCC_MIN - xi_lower[k])
            constraints_1.append(u1[k] <= 0.01 + xi_rate[k])
            constraints_1.append(xi_glare[k] == 0)

    obj_1 = cp.Minimize(
        10 * cp.sum(xi_upper) +
        10 * cp.sum(xi_lower) +
        100 * cp.sum(xi_glare) +
        1 * cp.sum(xi_rate)
    )
    
    prob_1 = cp.Problem(obj_1, constraints_1)
    
    try:
        prob_1.solve(solver=cp.GUROBI, verbose=False)
    except:
        try:
            prob_1.solve(solver=cp.CBC, verbose=False)
        except:
            return np.clip(a_RL, 0, 1), b_RL, None
            
    if prob_1.status not in ['optimal', 'optimal_inaccurate']:
        return np.clip(a_RL, 0, 1), b_RL, None
    
    xi_upper_locked = xi_upper.value + 1e-5
    xi_lower_locked = xi_lower.value + 1e-5
    xi_rate_locked = xi_rate.value + 1e-5
    xi_glare_locked = xi_glare.value + 1e-5
    
    stage1_u = u1.value
    stage1_blind = np.argmax(delta1[0].value)
    slack_glare_total = np.sum(xi_glare.value)
    slack_upper_total = np.sum(xi_upper.value)
    slack_lower_total = np.sum(xi_lower.value)
    slack_rate_total = np.sum(xi_rate.value)

    u2 = cp.Variable((N, n_lights))
    delta2 = cp.Variable((N, n_blinds), boolean=True)
    x2 = cp.Variable((N + 1, n_sensors))
    x2_vert = cp.Variable((N + 1, n_sensors))
    
    constraints_2 = []
    constraints_2.append(x2[0] == x_prev)
    
    for k in range(N):
        daylight_k = sum(delta2[k, m] * D[t + k + 1, m, :] for m in range(n_blinds))
        constraints_2.append(x2[k + 1] == L @ u2[k] + daylight_k)
        daylight_vert_k = sum(delta2[k, m] * D_vert[t + k + 1, m, :] for m in range(n_blinds))
        constraints_2.append(x2_vert[k + 1] == daylight_vert_k)
        constraints_2.append(cp.sum(delta2[k, :]) == 1)
        constraints_2.append(u2[k] >= 0)
        constraints_2.append(u2[k] <= 1)
        
        if k == 0:
            constraints_2.append(u2[k] - u_prev <= delta_max + xi_rate_locked[k])
            constraints_2.append(u2[k] - u_prev >= -delta_max - xi_rate_locked[k])
        else:
            constraints_2.append(u2[k] - u2[k-1] <= delta_max + xi_rate_locked[k])
            constraints_2.append(u2[k] - u2[k-1] >= -delta_max - xi_rate_locked[k])
        
        if occupancy[k]:
            if k < N - 1:
                constraints_2.append(x2[k + 1] <= PATH_MAX + xi_upper_locked[k])
                constraints_2.append(x2[k + 1] >= PATH_MIN - xi_lower_locked[k])
            else:
                constraints_2.append(x2[k + 1] <= TERM_MAX + xi_upper_locked[k])
                constraints_2.append(x2[k + 1] >= TERM_MIN - xi_lower_locked[k])
            constraints_2.append(x2_vert[k + 1] <= GLARE_MAX + xi_glare_locked[k])
        else:
            constraints_2.append(x2[k + 1] <= UNOCC_MAX + xi_upper_locked[k])
            constraints_2.append(x2[k + 1] >= UNOCC_MIN - xi_lower_locked[k])
            constraints_2.append(u2[k] <= 0.01 + xi_rate_locked[k])

    obj_2 = cp.Minimize(cp.sum_squares(u2[0] - a_RL) + w_blind * (1 - delta2[0, b_RL]))
    prob_2 = cp.Problem(obj_2, constraints_2)
    
    try:
        prob_2.solve(solver=cp.GUROBI, verbose=False)
    except:
        return np.clip(stage1_u[0], 0, 1), stage1_blind, {
            'correction_norm': np.linalg.norm(stage1_u[0] - a_RL),
            'slack_glare': slack_glare_total,
            'slack_upper': slack_upper_total,
            'slack_lower': slack_lower_total,
            'slack_rate': slack_rate_total,
        }

    if prob_2.status in ['optimal', 'optimal_inaccurate']:
        safe_u = np.clip(u2[0].value, 0, 1)
        safe_blind = np.argmax(delta2[0].value)
        # FIX: Include all slack values in diagnostics
        diagnostics = {
            'correction_norm': np.linalg.norm(safe_u - a_RL),
            'blind_changed': safe_blind != b_RL,
            'slack_glare': slack_glare_total,
            'slack_upper': slack_upper_total,
            'slack_lower': slack_lower_total,
            'slack_rate': slack_rate_total,
        }
        return safe_u, safe_blind, diagnostics
    else:
        # FIX: Include all slack values in fallback return
        return np.clip(stage1_u[0], 0, 1), stage1_blind, {
             'correction_norm': np.linalg.norm(stage1_u[0] - a_RL),
             'slack_glare': slack_glare_total,
             'slack_upper': slack_upper_total,
             'slack_lower': slack_lower_total,
             'slack_rate': slack_rate_total,
        }
