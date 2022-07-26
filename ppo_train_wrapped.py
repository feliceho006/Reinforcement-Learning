from math import fabs
from custom_policies.CustomCnn import CustomCNN
from utils import SaveOnBestTrainingRewardCallback
import wrappers
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import cv2
import matplotlib.pyplot as plt
import os
from gym.wrappers import RescaleAction, RecordVideo
from torch import tensor
import space_wrappers

# Create folders to save the results and logs
models_dir = "models/PPO/wrapped"
logdir = "logs"
temp_log_dir = "tmp/no_canny"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env_name = "CarRacing-v1"
env = gym.make(env_name)
env = Monitor(env, temp_log_dir)
env.reset()


# 1) GrayScale , 2) Blur, 3) Canny Edge Detector, 4) Crop
wrapped_env = wrappers.ObservationWrappers(env)

# 5) Action Wrapper
wrapped_env = space_wrappers.DiscretizedActionWrapper(wrapped_env, 3)

# Reward Wrapper
wrapped_env = wrappers.RewardWrapper(wrapped_env)


callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=temp_log_dir)
model = PPO("MlpPolicy", wrapped_env, ent_coef=0.01, verbose=1, tensorboard_log=logdir)
# model = PPO.load(f"{temp_log_dir}/best_model.zip", env=wrapped_env)

TIMESTEPS = 10000  # The number of env steps for each epoch
epochs = 100  # Number of training iterations

# Train the agent
for i in range(1, epochs):
    print("Starting epoch {}".format(i))
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO_wrapped_obs_action_discretized_uncanny",
        callback=callback,
    )
    if i % 3 == 0:
        model.save(f"{models_dir}/{TIMESTEPS*i}")


"""Modify the environment"""


# 2)


# 3) Action Wrapper

# print("Base action space: " , env.action_space)

# wrapped_env = RescaleAction(wrapped_env, min_action=0, max_action=0.3)
print("Wrapped action space: ", wrapped_env.action_space)


# Evaluate performance
episodes = 3
for ep in range(episodes):
    obs = wrapped_env.reset()
    accum_reward = 0
    done = False
    while not done:
        # action, _ = model.predict(obs)  # action is a numpy array
        action = wrapped_env.action_space.sample()

        print(action)

        obs, reward, done, info = wrapped_env.step(action)
        accum_reward += reward
        plt.imshow(obs, cmap="gray", vmin=0, vmax=255)
        plt.title("Reward so far: " + str(accum_reward) + " |  Episode: " + str(ep))
        plt.show()
        # wrapped_env.render()
wrapped_env.close()
