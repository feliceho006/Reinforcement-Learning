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
    print("loading model...")
    env = gym.make(ENV_ID)
    env = wrappers.ObservationWrappers(env)
    obs = env.reset()
    img = env.render(mode="rgb_array")
    model = PPO.load(f"{models_dir}/epoch_24.zip", env=env)
    print("model successfully loaded!")
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"Mean reward: {mean_reward}")
    print(f"Std dev reward: {std_reward}")

    images = []
    print("Beginning episodic iteration")
    for i in range(3000):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        img = env.render(mode="rgb_array")
    imageio.mimsave(
        "car_ppo_filb_wrapped.gif",
        [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
        fps=29,
    )
    print(".gif saved!")
