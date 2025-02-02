#!/usr/bin/env python
import gymnasium as gym
import os
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# Number of episodes
n_episodes = 500

video_dir = './video'

# Create environment and apply RecordVideo wrapper
env = gym.make("CartPole-v0", render_mode="rgb_array")

# Wrap environment to record videos of each episode
env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)  # Record each episode
env = RecordEpisodeStatistics(env)  # Track stats per episode

# Initialize variables to track performance across episodes
reward_list = []

# Loop for the number of episodes
for episode in tqdm(range(n_episodes)):
    total_reward = 0.0
    step = 0
    observation, info = env.reset(seed=42)
    
    # Run the episode with a deterministic policy (alternating left and right)
    action = 0  # Start with pushing the cart left (action 0)
    
    while True:
        # Toggle between left (0) and right (1) each timestep
        action = 1 if step % 2 != 0 else 0

        # Step in the environment
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
report_file = os.path.join(report_dir, 'Task3b_report.txt')

try:
    with open(report_file, 'w') as f:
        f.write(f"Task3b-CartPole-v0 Deterministic Agent Report\n")
        f.write(f"====================================\n")
        f.write(f"Number of episodes: {n_episodes}\n")
        f.write(f"Maximum reward: {max_reward}\n")
        f.write(f"Average reward: {avg_reward}\n")
        f.write(f"Standard deviation of reward: {std_reward}\n")
        f.write(f"====================================\n")
        f.write("\nNote: The agent used a deterministic policy alternating between left and right movements.\n")
    print(f"Report saved to {report_file}")
except Exception as e:
    print(f"Error while writing report: {e}")

# Close the environment
env.close()
