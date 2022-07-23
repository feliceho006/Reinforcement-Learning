from math import fabs
import wrappers
import gym
from stable_baselines3 import PPO
import cv2
import matplotlib.pyplot as plt
import os
from gym.wrappers import RescaleAction, RecordVideo
from torch import tensor 

# Create folders to save the results and logs
models_dir = "models/PPO/wrapped"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)



env_name = "CarRacing-v1" 
env = gym.make(env_name)
env.reset()



# 1) GrayScale , 2) Blur, 3) Canny Edge Detector, 4) Crop
wrapped_env = wrappers.ObservationWrappers(env)

model = PPO("MlpPolicy", wrapped_env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000 # The number of env steps for each epoch
epochs = 20  # Number of training iterations 

# Train the agent
for i in range(1, epochs):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_wrapped")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


"""Modify the environment"""



# 2) 


# 3) Action Wrapper

# print("Base action space: " , env.action_space)

# wrapped_env = RescaleAction(wrapped_env, min_action=0, max_action=0.3)
print("Wrapped action space: " , wrapped_env.action_space)

