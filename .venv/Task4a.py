#!/usr/bin/env python
#Random Policy: MountainCar-v0 environment
import gymnasium as gym
import os
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# Random Policy Environment: MountainCar-v0
def random_policy(env, n_episodes=10):
    video_dir = './video/Task4a'
    os.makedirs(video_dir, exist_ok=True)
    
    # Create environment and apply RecordVideo wrapper
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)  # Record each episode
    env = RecordEpisodeStatistics(env)  # Track stats per episode

    reward_list = []
    
    for episode in tqdm(range(n_episodes)):
        total_reward = 0.0
        step = 0
        observation, info = env.reset(seed=42)
        
        while True:
            # Take a random action
            action = env.action_space.sample()

            # Step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated:
                reward_list.append(total_reward)
                break
    
    env.close()
    return reward_list

# Running Random Policy in MountainCar-v0
if __name__ == "__main__":
    random_rewards = random_policy(gym.make("MountainCar-v0"), n_episodes=100)

    # Calculate and print statistics
    def print_statistics(rewards, policy_name):
        max_reward = np.max(rewards)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"\n{policy_name} Policy Stats:")
        print(f"Maximum reward: {max_reward}")
        print(f"Average reward: {avg_reward}")
        print(f"Standard deviation of reward: {std_reward}")
        return max_reward, avg_reward, std_reward

    max_reward, avg_reward, std_reward = print_statistics(random_rewards, "Random")

    # Report generation
    report_dir = './report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_file = os.path.join(report_dir, 'Task4a.txt')
    try:
        with open(report_file, 'w') as f:
            f.write(f"Task4a-Random Policy on MountainCar-v0 Report\n")
            f.write(f"====================================\n")
            f.write(f"Number of episodes: 100\n")
            f.write(f"Maximum reward: {max_reward}\n")
            f.write(f"Average reward: {avg_reward}\n")
            f.write(f"Standard deviation of reward: {std_reward}\n")
            f.write(f"====================================\n")
            f.write("\nNote: The agent used a random policy in the MountainCar-v0 environment.\n")
        print(f"Report saved to {report_file}")
    except Exception as e:
        print(f"Error while writing report: {e}")
