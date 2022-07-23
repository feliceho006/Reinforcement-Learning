import gym
from stable_baselines3 import PPO, A2C

env_name = "CarRacing-v1" 
env = gym.make(env_name)
env.reset()

# ask for user input
# model_name = input("Enter model name: ")
model_name = "PPO"
print("model name: ", model_name)
if model_name == "PPO":
    models_dir = "models/PPO"
    model_path = f"{models_dir}/70000.zip"
    model = PPO.load(model_path, env=env)
else:
    print("Invalid model name")
    # end the program if the model name is invalid
    model = None 
    
    
if model: 
    episodes = 10

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, _ = model.predict(obs) # action is a numpy array
            obs, reward, done, info = env.step(action)

    env.close()
 