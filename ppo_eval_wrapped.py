import gym
from stable_baselines3 import PPO, A2C
import wrappers
import space_wrappers

env_name = "CarRacing-v1" 
env = gym.make(env_name)
env.reset()

# 1) GrayScale , 2) Blur, 3) Canny Edge Detector, 4) Crop
wrapped_env = wrappers.ObservationWrappers(env)

# 5) Action Wrapper
wrapped_env = space_wrappers.DiscretizedActionWrapper(wrapped_env, 3)

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
    model_path = f"{models_dir}/best_model.zip"
    model = PPO.load(model_path, env=wrapped_env)
else:
    print("Invalid model name")
    # end the program if the model name is invalid
    model = None 
    
    
    
# Recording
# wrapped_env = gym.wrappers.RecordVideo(wrapped_env, 'video_1', episode_trigger = lambda x: x % 2 == 0)
# We can review the training performance for 10 episodes
episodes = 3
for ep in range(episodes):
    obs = wrapped_env.reset()
    # greyscale_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    done = False
    while not done:
        action, _ = model.predict(obs) # action is a numpy array
        # print(action)

        obs, reward, done, info = wrapped_env.step(action)

        # plt.imshow(obs.permute(1,2,0))  # (H, W, C)
        # plt.show()
        # .to("cpu").numpy())
        wrapped_env.render()
wrapped_env.close()
# wrapped_env.generate_video()
 