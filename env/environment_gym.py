import gym

class RocketMeister10(gym.Env):
    def __init__(self, env_config={}):
        pass       

    def reset(self):
        pass


    def reset_rocket_state(self, x=500, y=100, ang=1e-9, vel_x=0, vel_y=0, level=0):  # ang=1e-10
        pass

    def set_done(self):
        pass

    def step(self, action=[0, 0]):
        pass

    def render(self, mode=None):
        # initialize pygame only when render is called once
        import pygame
        import os
        from PIL import Image
        middle_echo_index = (self.rocket.N_ECHO - 1) // 2

        def init_renderer(self):
            pass
            

        def draw_level():
            pass

        def draw_goal_next():
            pass

        def draw_goal_all():
            pass

        def draw_rocket():
            pass

        def draw_goal_intersection_points():
            pass

        def draw_echo_collision_points():
            pass

        def draw_text(surface, text=None, size=30, x=0, y=0, 
                      font_name=pygame.font.match_font('consolas'), 
                      position='topleft'):
            pass
            
        def get_gui_value(value: str):
            pass

    def get_rocket_state(self):
        pass

    def update_rocket_state(self, rocket_state):
        pass

    def update_interface_vars(self, action_next):
        pass