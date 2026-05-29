# microgrid_env.py

import numpy as np
import scipy.io
from config import *

class MicrogridEnv:
    def __init__(self):
        # Load Simulink data
        data = scipy.io.loadmat(DATA_PATH)
        self.pv_profile   = data['PV_data'].flatten()[:NUM_HOURS]
        self.load_profile = data['Load_data'].flatten()[:NUM_HOURS]
        
        self.reset()

    def reset(self):
        self.hour = 0
        self.SoC  = SOC_INIT
        state = self._get_state()
        return state

    def _get_state(self):
        return np.array([
            self.SoC,
            self.pv_profile[self.hour],
            self.load_profile[self.hour],
            self.hour / NUM_HOURS       # normalized hour
        ], dtype=np.float32)

    def step(self, action):
        P_pv   = self.pv_profile[self.hour]
        P_load = self.load_profile[self.hour]
        P_net  = P_pv - P_load

        # Action to power command
        if action == 1 and self.SoC < SOC_MAX:
            P_cmd = min(MAX_CHARGE_POWER, max(0, P_net))
        elif action == 2 and self.SoC > SOC_MIN:
            P_cmd = -min(MAX_DISCHARGE_POWER, max(0, -P_net))
        else:
            P_cmd = 0

        # SoC update
        self.SoC = self.SoC + P_cmd / BATTERY_CAPACITY
        self.SoC = np.clip(self.SoC, SOC_MIN, SOC_MAX)

        # Grid import — unmet load
        P_grid = max(0, P_load - P_pv - (-P_cmd))

        # Reward
        reward = self._get_reward(P_grid, self.SoC)

        # Next step
        self.hour += 1
        done  = self.hour >= NUM_HOURS
        state = self._get_state() if not done else np.zeros(STATE_SIZE)

        return state, reward, done, {'P_grid': P_grid, 'SoC': self.SoC}

    def _get_reward(self, P_grid, SoC):
        # Penalize grid import
        grid_penalty = -P_grid * 2.0

        # Penalize SoC violations
        if SoC <= SOC_MIN:
            soc_penalty = -10.0
        elif SoC >= SOC_MAX:
            soc_penalty = -2.0
        else:
            soc_penalty = 0.0

        return grid_penalty + soc_penalty