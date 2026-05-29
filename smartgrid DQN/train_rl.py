# train_rl.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from microgrid_env import MicrogridEnv
from utils import plot_results, print_episode_stats, moving_average
from config import *

# ─── Neural Network ───────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# ─── DQN Agent ────────────────────────────────────────────────
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory    = deque(maxlen=MEMORY_SIZE)
        self.epsilon   = EPSILON_START
        self.steps     = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ─── Training Loop ────────────────────────────────────────────
def train():
    env   = MicrogridEnv()
    agent = DQNAgent()

    episode_rewards = []
    best_reward     = -np.inf

    for episode in range(NUM_EPISODES):
        state        = env.reset()
        total_reward = 0
        soc_history  = []
        grid_history = []

        for hour in range(NUM_HOURS):
            action                       = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            state         = next_state
            total_reward += reward
            soc_history.append(info['SoC'])
            grid_history.append(info['P_grid'])

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        episode_rewards.append(total_reward)

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

        if episode % 50 == 0:
            print_episode_stats(episode, total_reward, agent.epsilon, sum(grid_history))

    # Final plot
    plot_results(episode_rewards, soc_history, grid_history)
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")

if __name__ == '__main__':
    train()