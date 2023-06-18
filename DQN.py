import torch
from torch import nn
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA=0.99  # discount rate
BATCH_SIZE=32   # how many transitions we sample from the replay buffer
BUFFER_SIZE=50000   # maximum number of transitions stored
MIN_REPLAY_SIZE=1000    # number of transitions to start computing gradient and training
EPSILON_START=1.0   # initial value of epsilon
EPSILON_END=0.02    # final value of epsilon
EPSILON_DECAY=10000     # number of steps to decay from initial value to final value of epsilon
TARGET_UPDATE_FREQ=1000     # number of steps to set the target parameters equal to the online parameters
LR=5e-4

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        in_features = int(np.prod(env.observation_space.shape))
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self(obs_tensor.unsqueeze(0))
        
        max_q_index = torch.argmax(q_value, dim=1)[0]
        action = max_q_index.detach().item()
        
        return action

env = gym.make('CartPole-v0', render_mode='human')
replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

# initialize Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    
    new_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    
    if done:
        obs = env.reset()

# Main Training Loop
obs = env.reset()[0]

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    
    random_sample = random.random()
    
    if random_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
    
    new_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    
    episode_reward += reward
    
    if done:
        obs = env.reset()[0]
        
        reward_buffer.append(episode_reward)
        episode_reward = 0.0
    
    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)
    
    obses = []
    for t in transitions:
        if (len(t[0]) == 2):
            obses.append(t[0][0])
        else:
            obses.append(t[0])

    # obses = [t[0] for t in transitions]
    # obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])
    
    obses_tensor = torch.as_tensor(obses, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_tensor = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_tensor = torch.as_tensor(new_obses, dtype=torch.float32)
    
    # Compute Targets
    target_q_values = target_net(new_obses_tensor)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    
    targets = rewards_tensor + GAMMA * (1 - dones_tensor) * max_target_q_values
    
    # Compute Loss
    q_values = online_net(obses_tensor)
    
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_tensor)
    
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    
    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Reward', np.mean(reward_buffer))
