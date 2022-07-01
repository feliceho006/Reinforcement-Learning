from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):

    @abstractmethod
    def __init__(self, game):
        self.game = game
        self.load_level()

    @abstractmethod
    def load_level(self):
        #level definition
        pass
    
    @abstractmethod
    def set_level_vectors(self, line1, line2, goals, level_collision_vectors = None):
        # list for pygame draw
        pass

    @abstractmethod
    def generate_level_vectors(self):
        pass