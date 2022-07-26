import os
from math import fabs

import cv2
import gym
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo, RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch import tensor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import space_wrappers
import wrappers
from custom_policies.CustomCnn import CustomCNN
from space_wrappers.misc import StackObservationWrapper
from utils import SaveOnBestTrainingRewardCallback

# Create folders to save the results and logs
models_dir = "models/PPO/wrapped_stacked"
logdir = "logs"
temp_log_dir = "tmp/stacked"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env_name = "CarRacing-v1"
env = gym.make(env_name)
env = Monitor(env, temp_log_dir)
env.reset()



def make_vec_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = wrappers.ObservationWrappers(env)
        env = StackObservationWrapper(env, 4, 0)
        # wrapped_env = space_wrappers.DiscretizedActionWrapper(wrapped_env, 3)
        env = wrappers.RewardWrapper(env)
        return env

    set_random_seed(seed)
    return _init

"""
PARAMS
"""
NO_ENVS = 4
ENV_ID = "CarRacing-v1"
OPT_EPOCHS = 10




# wrapped_env = SubprocVecEnv([make_vec_env(ENV_ID, i) for i in range(NO_ENVS)])

wrapped_env = make_vec_env(ENV_ID, 1)()

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=temp_log_dir)

model = PPO("MlpPolicy", wrapped_env, n_epochs=OPT_EPOCHS, ent_coef=0.01, verbose=1, tensorboard_log=logdir)



TIMESTEPS = 10000  # The number of env steps for each epoch
epochs = 100  # Number of training iterations

# Train the agent
for i in range(1, epochs):
    print("Starting epoch {}".format(i))
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO_wrapped_obs_action_discretized_stacked",
        callback=callback,
    )
    if i % 3 == 0:
        model.save(f"{models_dir}/{TIMESTEPS*i}")


# Evaluate performance
episodes = 3
for ep in range(episodes):
    obs = wrapped_env.reset()
    accum_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)  # action is a numpy array
        # action = wrapped_env.action_space.sample()

        print(action)

        obs, reward, done, info = wrapped_env.step(action)
        accum_reward += reward
        # plt.imshow(obs[0], cmap="gray", vmin=0, vmax=255)
        # plt.title("Reward so far: " + str(accum_reward) + " |  Episode: " + str(ep))
        # plt.show()
        wrapped_env.render()
wrapped_env.close()
