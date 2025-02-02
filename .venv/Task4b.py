#!/usr/bin/env python
#Deterministic Policy: LunarLander-v3 environment
import gymnasium as gym
import os
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# Deterministic Policy Environment: LunarLander-v3
def deterministic_policy(env, n_episodes=10):
    video_dir = './video/Task4b'
    os.makedirs(video_dir, exist_ok=True)
    
    # Create environment and apply RecordVideo wrapper
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)  # Record each episode
    env = RecordEpisodeStatistics(env)  # Track stats per episode

    reward_list = []
    
    for episode in tqdm(range(n_episodes)):
        total_reward = 0.0
        step = 0
        observation, info = env.reset(seed=42)
        
        while True:
            # Use a deterministic policy to control the lander
            x, y, vx, vy, angle, angular_velocity, left_thrust, right_thrust = observation
            # Simple deterministic policy: adjust thrusters based on the lander's position
            if angle < 0:
                action = 2  # Fire left thrust
            elif angle > 0:
                action = 3  # Fire right thrust
            else:
                if vy < -0.1:
                    action = 1  # Fire main engine (downward thrust)
                else:
                    action = 0  # No action (idle)
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated:
                reward_list.append(total_reward)
                break
    
    env.close()
    return reward_list

# Running Deterministic Policy in LunarLander-v2
if __name__ == "__main__":
    deterministic_rewards = deterministic_policy(gym.make("LunarLander-v3"), n_episodes=100)

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

    max_reward, avg_reward, std_reward = print_statistics(deterministic_rewards, "Deterministic")

    # Report generation
    report_dir = './report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_file = os.path.join(report_dir, 'Task4b.txt')
    try:
        with open(report_file, 'w') as f:
            f.write(f"Task4b-Deterministic Policy on LunarLander-v3 Report\n")
            f.write(f"====================================\n")
            f.write(f"Number of episodes: 100\n")
            f.write(f"Maximum reward: {max_reward}\n")
            f.write(f"Average reward: {avg_reward}\n")
            f.write(f"Standard deviation of reward: {std_reward}\n")
            f.write(f"====================================\n")
            f.write("\nNote: The agent used a deterministic policy in the LunarLander-v3 environment.\n")
        print(f"Report saved to {report_file}")
    except Exception as e:
        print(f"Error while writing report: {e}")
