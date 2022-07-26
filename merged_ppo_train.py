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
models_dir = "models/PPO/wrapped_vectorised"
logdir = "logs"
temp_log_dir = "tmp/vectorised"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env_name = "CarRacing-v1"
env = gym.make(env_name)
env = Monitor(env, temp_log_dir)
env.reset()

wrapped_env = wrappers.ObservationWrappers(env)
