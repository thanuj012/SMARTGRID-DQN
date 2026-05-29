# utils.py

import numpy as np
import matplotlib.pyplot as plt
import os
from config import *

def plot_results(episode_rewards, soc_history, grid_history, save=True):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Episode rewards
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True)

    # SoC over 24hrs (last episode)
    axes[1].plot(range(NUM_HOURS), soc_history)
    axes[1].axhline(y=SOC_MAX, color='r', linestyle='--', label='Max SoC')
    axes[1].axhline(y=SOC_MIN, color='b', linestyle='--', label='Min SoC')
    axes[1].set_title('Battery SoC - Last Episode')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('SoC')
    axes[1].legend()
    axes[1].grid(True)

    # Grid import over 24hrs (last episode)
    axes[2].bar(range(NUM_HOURS), grid_history)
    axes[2].set_title('Grid Import - Last Episode')
    axes[2].set_xlabel('Hour')
    axes[2].set_ylabel('Power (kW)')
    axes[2].grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_PATH + 'results.png')
    plt.show()

def print_episode_stats(episode, reward, epsilon, grid_total):
    print(f"Episode {episode:4d} | "
          f"Reward: {reward:8.2f} | "
          f"Epsilon: {epsilon:.3f} | "
          f"Grid Import: {grid_total:.2f} kWh")

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')