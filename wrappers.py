import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torchvision import transforms as T
import cv2
import random
import matplotlib.pyplot as plt


class ObservationWrappers(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.frames = []

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def blur_image(self, observation):
        blur = cv2.GaussianBlur(observation, (5, 5), 0)
        return blur

    def observation(self, observation):
        # observation = self.permute_orientation(observation)
        # transform = T.Grayscale()
        # observation = transform(observation)
        # cropped = observation[63:80, 24:73]
        cropped = observation[:-12]

        # fill the bottom of the image with black
        
        cropped = cv2.copyMakeBorder(cropped, 0, 12, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # plt.imshow(cropped)
        # plt.show()
        # cropped = observation
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        blur = self.blur_image(gray)
        canny = self.canny_edge_detector(blur)

        self.frames.append(canny)
        # self.video.write(canny)
        # plt.imshow(canny, cmap="gray", vmin=0, vmax=255)
        # plt.show()
        return canny

    def canny_edge_detector(self, observation):
        canny = cv2.Canny(observation, 50, 150)
        return canny

    def generate_video(self):
        print("generating video")
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        self.video = cv2.VideoWriter("./video.avi", fourcc, 20, (17, 49))

        for i in range(len(self.frames)):
            self.video.write(self.frames[i])
        self.video.release()


# class ObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def observation(self, obs):
#         # Normalise observation by 255
#         return obs / 255.0


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = self.permute_orientation(observation)
        observation = transforms(observation).squeeze(0)
        return observation


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return [1]
        if action == 3:
            return random.choice([0, 1, 2])
        else:
            return action

# reduce the action space with a wrapper
class DiscreteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.__actions__ = [(-1, 0.2, 0.), (0, 0.2, 0.), (1, 0.2, 0.),
                            (-1, 0.5, 0.), (0, 0.5, 0.), (1, 0.5, 0.), #           Action Space Structure
                            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
                            (-1, 0, 0.8), (0, 0, 0.8), (1, 0, 0.8), # Range        -1~1       0~1   0~1
                            (-1, 0, 0.3), (0, 0, 0.3), (1, 0, 0.3),
                            (-1, 0,   0), (0, 0,   0), (1, 0,   0)]

        
        self.action_space = Discrete(len(self.__actions__)-1)
        
    def step(self, action):
        print(action)
        next_state, reward, done, info = self.env.step(self.__actions__[action])
        # modify ...
        return next_state, reward, done, info