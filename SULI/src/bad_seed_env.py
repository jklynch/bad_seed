import numpy as np
import gym
from gym import spaces
from random import random

COUNT = 0
SAMPLES = 5
TRIALS = 10
grid = []


class BadSeedEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code

    def __init__(self, grid_size_x=TRIALS, grid_size_y=SAMPLES):
        super(BadSeedEnv, self).__init__()

        # Size of the 2D-grid
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        for i in range(grid_size_y):
            col = []
            for j in range(grid_size_x):
                col.append(0)
            grid.append(col)

            # Initialize the agent at the right of the grid
        self.agent_pos = [0, 0]

        # Define action and observation space
        # They must be gym.spaces objects
        # The number of actions will beb how many samples there are (lets start w 5 and generalize later)
        n_actions = SAMPLES
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        # WHAT IS THE DIFFERECE BETWEEN LOW/HIGH AND SHAPE
        self.observation_space = spaces.Box(low=0, high=self.grid_size_x,
                                            shape=(SAMPLES, self.grid_size_x,), dtype=np.uint8)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, action):
        if action >= 0 and action < SAMPLES:
            self.agent_pos[action] = random()
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size_x)

        # Are we at the left of the grid?
        done = bool(self.agent_pos == 0)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size_x - self.agent_pos))

    def close(self):
        pass
