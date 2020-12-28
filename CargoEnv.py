import numpy as np
import gym

from gym.envs.toy_text import discrete
from gym.envs.classic_control import cartpole

direction = {
    0: 'Right',
    1: 'Left',
    2: 'Down',
    3: 'Up',
}


class CargoEnv:
    def __init__(self, width: int = 3, height: int = 1, tell_carry_state: bool = True, random_start: bool = False,
                 max_steps=50, delivery_reward=21, non_delivery_reward=0, pick_up_reward=0, done_on_delivery=False,
                 seed: int = 10):
        # environment dimensions
        self.random_seed = seed
        self.done_on_delivery = done_on_delivery
        self.name = 'CargoEnv'
        self.width = width
        self.height = height
        self.tell_carry_state = tell_carry_state
        self.random_start = random_start
        self.max_step = max_steps
        self.non_delivery_reward = non_delivery_reward
        self.delivery_reward = delivery_reward
        self.pick_up_reward = pick_up_reward

        # base class setup
        self.nS = self.width * self.height
        self.nA = 4 if self.height > 1 else 2

        # current state
        self.carry = False
        self.step_count = 0
        self.debug_data = None
        if not self.random_start:
            self.x_pos = int(self.width / 2)
            self.y_pos = int(self.height / 2)
        else:
            self.x_pos = np.random.randint(1, self.width)
            self.y_pos = np.random.randint(0, self.height + 1)

        # ovv and action spaces
        self.action_space = gym.spaces.Discrete(self.nA)

    def reset(self):
        self.carry = False
        self.step_count = 0
        self.debug_data = None
        if not self.random_start:
            self.x_pos = int(self.width / 2)
            self.y_pos = int(self.height / 2)
        else:
            self.x_pos = np.random.randint(1, self.width)
            self.y_pos = np.random.randint(0, self.height + 1)
        return self.x_pos, self.y_pos, 0

    def constraint_pos(self):
        if self.y_pos >= self.height:
            self.y_pos = self.height - 1
        if self.y_pos < 0:
            self.y_pos = 0
        if self.x_pos >= self.width:
            self.x_pos = self.width - 1
        if self.x_pos < 0:
            self.x_pos = 0

    def step(self, a):
        if not self.action_space.contains(a):
            raise Exception("unknown action {}".format(int(a)))
        if a == 0:
            self.x_pos += 1
        elif a == 1:
            self.x_pos -= 1
        elif a == 2:
            self.y_pos += 1
        elif a == 3:
            self.y_pos -= 1
        self.constraint_pos()
        self.step_count += 1

        # checking end condition
        if self.step_count == self.max_step:
            d = True
        else:
            d = False

        # calculating reward
        r = -1
        if self.x_pos is 0 and self.y_pos is int(self.height / 2):
            if self.carry:
                r += self.delivery_reward
                self.carry = False
                if self.done_on_delivery:
                    d = True
            else:
                r += self.non_delivery_reward
        if self.x_pos is self.width - 1 and self.y_pos is int(self.height / 2):
            if not self.carry:
                r += self.pick_up_reward
                self.carry = True

        # return state
        s = (self.x_pos, self.y_pos, int(self.carry and self.tell_carry_state))
        self.debug_data = (direction[a], r, self.carry)
        return s, r, d, {"prob": "place holder"}

    def render(self, mode):
        screen = np.zeros(shape=(self.height, self.width))
        screen[self.y_pos, self.x_pos] = 1
        print('{}  ({}, Step - {})'.format(
            screen,
            self.debug_data,
            self.step_count
        ), end='\n\n')
        return screen

    def seed(self, seed: int):
        self.random_seed = seed
        np.random.seed(seed)
        return seed
