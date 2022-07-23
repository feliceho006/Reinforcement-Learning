import gym
from stable_baselines3 import PPO, A2C
import wrappers

env_name = "CarRacing-v1" 
env = gym.make(env_name)
env.reset()

env = wrappers.ObservationWrappers(env)



# ask for user input
# model_name = input("Enter model name: ")
model_name = "PPO_wrapped"
print("model name: ", model_name)
if model_name == "PPO_wrapped":
    models_dir = "models/PPO/wrapped"
    model_path = f"{models_dir}/20000.zip"
    model = PPO.load(model_path, env=env)
else:
    print("Invalid model name")
    # end the program if the model name is invalid
    model = None 
    
    
    
# Recording
env = gym.wrappers.RecordVideo(env, '20000_video', episode_trigger = lambda x: x % 2 == 0)
# We can review the training performance for 10 episodes
episodes = 3
for ep in range(episodes):
    obs = env.reset()
    # greyscale_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    done = False
    while not done:
        action, _ = model.predict(obs) # action is a numpy array
        print(action)

        obs, reward, done, info = env.step(action)

        # plt.imshow(obs.permute(1,2,0))  # (H, W, C)
        # plt.show()
        # .to("cpu").numpy())
        env.render()
env.close()
# wrapped_env.generate_video()
 