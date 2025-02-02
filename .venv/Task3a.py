#!/usr/bin/env python
import gymnasium as gym
import math
import imageio.v2 as imageio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import load, dump
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# number of episodes
n_episodes = 500

video_dir = './video'

#Create Env and apply RecordVideo wrapper
env=gym.make("CartPole-v0", render_mode="rgb_array")

#wrap env to record videos of each episode
env = RecordVideo(env, video_dir, episode_trigger=lambda e: True) #Record each episode
env = RecordEpisodeStatistics(env) #to Track stats per episode

# Initialize variables to track performance across episodes
reward_list = []

# Loop for the number of episodes
for episode in tqdm(range(n_episodes)):
    total_reward = 0.0
    step = 0
    observation, info = env.reset(seed=42)
    
    while True:
        # Take random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if terminated:
            reward_list.append(total_reward)
            break

# Calculate statistics on total rewards
max_reward = np.max(reward_list)
avg_reward = np.mean(reward_list)
std_reward = np.std(reward_list)

# Ensure the report directory exists
report_dir = './report'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)
    print(f"Report directory created at: {report_dir}")

# Define the report file path
report_file = os.path.join(report_dir, 'Task3a_report.txt')

try:
    with open(report_file, 'w') as f:
        f.write(f"Task3a-CartPole-v0 Random Agent Report\n")
        f.write(f"====================================\n")
        f.write(f"Number of episodes: {n_episodes}\n")
        f.write(f"Maximum reward: {max_reward}\n")
        f.write(f"Average reward: {avg_reward}\n")
        f.write(f"Standard deviation of reward: {std_reward}\n")
        f.write(f"====================================\n")
        f.write("\nNote: The agent used random actions. This report reflects the performance of a random agent on the CartPole-v0 environment.\n")
    print(f"Report saved to {report_file}")
except Exception as e:
    print(f"Error while writing report: {e}")

env.close()


