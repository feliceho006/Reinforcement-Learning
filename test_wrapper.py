import gym
from stable_baselines3 import PPO, A2C
import wrappers
import space_wrappers

env_name = "CarRacing-v1" 
env = gym.make(env_name)
env.reset()

env = wrappers.ObservationWrappers(env)

# env = wrappers.ActionWrapper(env)
env = space_wrappers.DiscretizedActionWrapper(env, 3)

episodes = 3
for ep in range(episodes):
    obs = env.reset()
    # greyscale_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    done = False
    while not done:
        # action, _ = model.predict(obs) # action is a numpy array
        action = env.action_space.sample()
        print(action)

        # obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)

        # plt.imshow(obs.permute(1,2,0))  # (H, W, C)
        # plt.show()
        # .to("cpu").numpy())
        env.render()
env.close()