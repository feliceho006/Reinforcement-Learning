from cgitb import reset
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import wrappers
import os

"""
HELPER FUNCTIONS
"""


def make_vec_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = wrappers.ObservationWrappers(env)
        return env

    set_random_seed(seed)
    return _init


"""
DIRECTORIES
"""
# Create folders to save the results and logs
models_dir = "models/PPO/wrapped-filb/"
logdir = "logs/ppo_filb"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


"""
PARAMS
"""
NO_ENVS = 4
ENV_ID = "CarRacing-v1"
OPT_EPOCHS = 10
NO_EPOCHS = 25
TIMESTEPS = 3e3
if __name__ == "__main__":
    """
    ENV PROCESSING
    """
    #! Vectorizing: trains agent on 4 environments at a time
    env = SubprocVecEnv([make_vec_env(ENV_ID, i) for i in range(NO_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # env = make_vec_env(ENV_ID, n_envs=NO_ENVS, seed=0, vec_env_cls=SubprocVecEnv)
    # wrapped_env = vectorized_wrappers.ObservationWrappers(env)
    # print("isunwrap: ", env.env_is_wrapped(wrappers.ObservationWrappers))
    """
    MODEL
    """
    model = PPO(
        policy="MlpPolicy", env=env, n_epochs=OPT_EPOCHS, tensorboard_log=logdir
    )
    print("Training model...")
    for i in range(NO_EPOCHS):
        print(f"Beginning Epoch {i}")
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO Wrapped",
        )
        print(f"Epoch {i} complete. Saving model...")
        model.save(f"{models_dir}/epoch_{i}.zip")
        # # TODO: implement HER
    print("Training complete!")
    env.close()
