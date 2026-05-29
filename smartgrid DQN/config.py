# config.py

# Environment
NUM_HOURS = 24
BATTERY_CAPACITY = 50      # kWh
MAX_CHARGE_POWER = 5       # kW
MAX_DISCHARGE_POWER = 5    # kW
SOC_MIN = 0.2
SOC_MAX = 0.9
SOC_INIT = 0.5

# DQN Agent
STATE_SIZE = 4             # [SoC, P_pv, P_load, hour]
ACTION_SIZE = 3            # 0=idle, 1=charge, 2=discharge
LEARNING_RATE = 0.001
GAMMA = 0.99               # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01

MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON_DECAY = 0.9995   # slower decay, was 0.995
NUM_EPISODES  = 2000     # more episodes, was 1000

# Training

TARGET_UPDATE_FREQ = 10    # update target network every N episodes

# Paths
DATA_PATH = r'D:\projects and carrer\smartgrid DQN\microgrid_data.mat'
MODEL_PATH = 'dqn_model.pth'
RESULTS_PATH = 'results/'