from math import fabs
import gym
from stable_baselines3 import PPO
import os

from torch import tensor 

# Create folders to save the results and logs
models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)



env_name = "CarRacing-v1" 
env = gym.make(env_name)
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)



TIMESTEPS = 10000 # The number of env steps for each epoch
epochs = 20  # Number of training iterations 

# Train the agent
for i in range(1, epochs):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# We can review the training performance for 10 episodes
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:

        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())

env.close()
 