import gym
from stable_baselines3 import PPO, A2C
from space_wrappers.misc import StackObservationWrapper
import wrappers
import space_wrappers
import imageio
import numpy as np 
import keyboard

save_gif = False 
env_name = "CarRacing-v1"
env = gym.make(env_name).env
# print(env)
env.reset()


# 1) GrayScale , 2) Blur, 3) Canny Edge Detector, 4) Crop
wrapped_env = wrappers.ObservationWrappers(env)


# 5) Action Wrapper
wrapped_env = space_wrappers.DiscretizedActionWrapper(wrapped_env, 3)


# Stack observations
wrapped_env = StackObservationWrapper(wrapped_env, 4, 1)

# Reward Wrapper
wrapped_env = wrappers.RewardWrapper(wrapped_env)


# ask for user input
# model_name = input("Enter model name: ")
model_name = "PPO_wrapped"
print("model name: ", model_name)
if model_name == "PPO_wrapped":
    # models_dir = "models/PPO/wrapped/1"
    # model_path = f"{models_dir}/390000.zip"
    models_dir = "tmp"
    model_path = f"{models_dir}/wrapped_stacked_discrete/4/best_model.zip"
    model = PPO.load(model_path, env=wrapped_env)
else:
    print("Invalid model name")
    # end the program if the model name is invalid
    model = None





if save_gif:
    obs = wrapped_env.reset()

    img = wrapped_env.render(mode="rgb_array")

    images = []
    print("Beginning episodic iteration")
    for i in range(3000):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = wrapped_env.step(action)
        img = wrapped_env.render(mode="rgb_array")
    imageio.mimsave(
        "car_ppo_cayden_wrapped.gif",
        [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
        fps=29,
    )
    print(".gif saved!")

else:
    # Recording
    # wrapped_env = gym.wrappers.RecordVideo(wrapped_env, 'video_1', episode_trigger = lambda x: x % 2 == 0)
    # We can review the training performance for 10 episodes
    
    episodes = 3
    for ep in range(episodes):
        obs = wrapped_env.reset()

        done = False
        while not done:
            action, _ = model.predict(obs)  # action is a numpy array
            # print(action)

            obs, reward, done, info = wrapped_env.step(action)

            
            wrapped_env.render()
            # check for keyboard n press 
            if keyboard.is_pressed('q'):
                done = True
    wrapped_env.close()
    # print (wrapped_env)
    # print(wrapped_env.env)

