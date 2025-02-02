#!/usr/bin/env python
import gymnasium as gym
import math
import imageio.v2 as imageio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import load, dump

# number of episodes
n_episodes = 500

render_mode = 'rgb_array'  # or, 'human'
env = gym.make("CartPole-v0", render_mode=render_mode)
# Initialize empty buffer for the images that will be stitched into a gif
# Create a temp directory
filenames = []
try:
    os.mkdir("./temp")
except:
    pass

# Initialize variables to track performance across episodes
reward_list = []

# Loop for the number of episodes
for episode in tqdm(range(n_episodes)):
    total_reward = 0.0
    step = 0
    observation, info = env.reset(seed=42)
    
    # Create a gif for the episode
    episode_filenames = []

    #Initial action is pushing cart left (action 0)
    action = 0
    
    while True:
        # Plot the previous state and save it as an image that will be later patched together as a .gif
        if step % 5 == 0:
            img = plt.imshow(env.render())
            plt.title("Episode: {}, Step: {}".format(episode, step))
            plt.axis('off')
            plt.savefig("./temp/{}.png".format(step))
            plt.close()
            episode_filenames.append("./temp/{}.png".format(step))

        # Take the action (toggle between left and right)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Toggle action between left (0) and right (1)
        action = 1 - action  # If action was 0, change to 1; if action was 1, change to 0.

        step += 1

        if terminated:
            reward_list.append(total_reward)
            break

    # Add the filenames to the global list
    filenames.extend(episode_filenames)

# Stitch the images together to produce a .gif
with imageio.get_writer('./video/Task2-random.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup the images for the next run
for f in filenames:
    if os.path.exists(f):
        os.remove(f)


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
report_file = os.path.join(report_dir, 'Task2_report.txt')

try:
    with open(report_file, 'w') as f:
        f.write(f"Task2-CartPole-v0 Deterministic Policy  Report\n")
        f.write(f"====================================\n")
        f.write(f"Number of episodes: {n_episodes}\n")
        f.write(f"Maximum reward: {max_reward}\n")
        f.write(f"Average reward: {avg_reward}\n")
        f.write(f"Standard deviation of reward: {std_reward}\n")
        f.write(f"====================================\n")
        f.write("\nNote: The agent used Deterministic Policy. agent begins with pushing the cart left, and each time-step agent toggles the movement until the pole is dropped, i.e., the episode gets ended\n")
    print(f"Report saved to {report_file}")
except Exception as e:
    print(f"Error while writing report: {e}")

env.close()


