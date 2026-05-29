# baseline.py

import numpy as np
from microgrid_env import MicrogridEnv
from utils import plot_results
from config import *

def rule_based_policy(state):
    SoC    = state[0]
    P_pv   = state[1]
    P_load = state[2]
    P_net  = P_pv - P_load

    if P_net > 0 and SoC < SOC_MAX:
        return 1  # charge
    elif P_net < 0 and SoC > SOC_MIN:
        return 2  # discharge
    else:
        return 0  # idle

def run_baseline():
    env = MicrogridEnv()
    state = env.reset()

    soc_history  = []
    grid_history = []
    total_reward = 0

    for hour in range(NUM_HOURS):
        action = rule_based_policy(state)
        state, reward, done, info = env.step(action)

        soc_history.append(info['SoC'])
        grid_history.append(info['P_grid'])
        total_reward += reward

    print("\n--- Rule-Based Baseline Results ---")
    print(f"Total Reward     : {total_reward:.2f}")
    print(f"Total Grid Import: {sum(grid_history):.2f} kWh")
    print(f"Final SoC        : {soc_history[-1]:.3f}")
    print(f"SoC History      : {[round(s,2) for s in soc_history]}")
    print(f"Grid History     : {[round(g,2) for g in grid_history]}")

    plot_results([total_reward], soc_history, grid_history)

    return total_reward, soc_history, grid_history

if __name__ == '__main__':
    run_baseline()