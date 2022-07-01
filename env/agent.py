from abc import ABC, abstractmethod
import numpy as np

class RocketAgent(ABC):
    VEL_MAX = 15  # not used
    ROT_VEL = 0.08
    ACCELERATION = 0.5
    N_ECHO = 7  # must be odd

    def __init__(self, game, env):
        self.game = game
        self.env = env
        self.visible = True
        self.reset_game_state()

    @abstractmethod
    def reset_game_state(self, x=500, y=100, ang=0, vel_x=0, vel_y=0, level=0):
        pass
    
    @abstractmethod
    def update_state(self, rocket_state):
        pass

    @abstractmethod
    def rotate(self, rotate):  # input: action1
        pass
    
    @abstractmethod
    def accelerate(self, accelerate):  # input: action0
        pass

    @abstractmethod
    def move(self, action):
        pass
