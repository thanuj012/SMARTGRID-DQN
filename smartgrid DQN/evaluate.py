# evaluate.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from microgrid_env import MicrogridEnv
from train_rl import DQN, DQNAgent
from baseline import rule_based_policy
from config import *

def run_dqn():
    env   = MicrogridEnv()
    agent = DQNAgent()
    agent.policy_net.load_state_dict(torch.load(MODEL_PATH))
    agent.policy_net.eval()
    agent.epsilon = 0  # no exploration, pure exploitation

    state        = env.reset()
    soc_history  = []
    grid_history = []
    total_reward = 0

    for hour in range(NUM_HOURS):
        action                         = agent.select_action(state)
        state, reward, done, info      = env.step(action)
        soc_history.append(info['SoC'])
        grid_history.append(info['P_grid'])
        total_reward += reward

    return total_reward, soc_history, grid_history

def run_baseline():
    env   = MicrogridEnv()
    state = env.reset()

    soc_history  = []
    grid_history = []
    total_reward = 0

    for hour in range(NUM_HOURS):
        action                    = rule_based_policy(state)
        state, reward, done, info = env.step(action)
        soc_history.append(info['SoC'])
        grid_history.append(info['P_grid'])
        total_reward += reward

    return total_reward, soc_history, grid_history

def plot_comparison():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    hours = range(NUM_HOURS)

    dqn_reward,      dqn_soc,      dqn_grid      = run_dqn()
    baseline_reward, baseline_soc, baseline_grid = run_baseline()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('SMARTGRID-DQN vs Rule-Based Baseline', fontsize=14)

    # SoC comparison
    axes[0].plot(hours, dqn_soc,      label=f'DQN      (reward={dqn_reward:.2f})',      color='blue')
    axes[0].plot(hours, baseline_soc, label=f'Baseline (reward={baseline_reward:.2f})', color='orange', linestyle='--')
    axes[0].axhline(y=SOC_MAX, color='red',   linestyle=':', label='Max SoC')
    axes[0].axhline(y=SOC_MIN, color='green', linestyle=':', label='Min SoC')
    axes[0].set_title('Battery SoC over 24 Hours')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('SoC')
    axes[0].legend()
    axes[0].grid(True)

    # Grid import comparison
    axes[1].bar([h - 0.2 for h in hours], dqn_grid,      width=0.4, label=f'DQN      ({sum(dqn_grid):.2f} kWh)',      color='blue')
    axes[1].bar([h + 0.2 for h in hours], baseline_grid, width=0.4, label=f'Baseline ({sum(baseline_grid):.2f} kWh)', color='orange')
    axes[1].set_title('Grid Import over 24 Hours')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Power (kW)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'comparison.png')
    plt.show()

    print("\n--- Final Comparison ---")
    print(f"DQN      | Reward: {dqn_reward:.2f} | Grid Import: {sum(dqn_grid):.2f} kWh")
    print(f"Baseline | Reward: {baseline_reward:.2f} | Grid Import: {sum(baseline_grid):.2f} kWh")
    improvement = ((baseline_reward - dqn_reward) / abs(baseline_reward)) * 100
    print(f"Improvement: {improvement:.1f}% better reward")

if __name__ == '__main__':
    plot_comparison()