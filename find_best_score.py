import os
import gym
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wrappers

"""
DIRECTORIES
"""
# Create folders to save the results and logs
models_dir = "models/PPO/wrapped-filb/"
logdir = "logs/ppo_filb"
"""
CONSTS
"""
ENV_ID = "CarRacing-v1"


if __name__ == "__main__":
    """
    EVAL
    """
    env = gym.make(ENV_ID)
    env = wrappers.ObservationWrappers(env)
    obs = env.reset()
    img = env.render(mode="rgb_array")
    with open("result.txt", "w") as f:
        for i in range(24):
            print(f"Loading model for epoch {i}...")
            model = PPO.load(f"{models_dir}/epoch_{i}.zip", env=env)
            print("model successfully loaded!")
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), n_eval_episodes=5
            )
            print(f"Epoch {i}: Mean reward: {mean_reward}")
            print(f"Epoch {i}: Std dev reward: {std_reward}")
            f.write(f"Epoch {i}")
            f.write(f"Mean reward: {mean_reward}")
            f.write(f"Std dev reward: {std_reward}")
