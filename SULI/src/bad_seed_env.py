import numpy as np
import gym
from gym import spaces
from random import random
from tensorforce.environments import Environment

# from SULI.src.a2c_test import Model

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

    def __init__(self, grid_size_x=TRIALS, grid_size_y=SAMPLES, count = COUNT):
        super(BadSeedEnv, self).__init__()

        # Size of the 2D-grid
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.count = count
        for i in range(grid_size_y):
            col = []
            for j in range(grid_size_x):
                col.append(0)
            grid.append(col)

        # Define action and observation space
        # They must be gym.spaces objects
        # The number of actions will beb how many samples there are (lets start w 5 and generalize later)
        n_actions = SAMPLES
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        # WHAT IS THE DIFFERECE BETWEEN LOW/HIGH AND SHAPE
        self.observation_space = spaces.Box(low=0, high=TRIALS, shape=(SAMPLES, TRIALS), dtype=np.uint8)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.count = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.count]).astype(np.float32)

    def step(self, action):
        if action >= 0 and action < SAMPLES:
            grid[action][self.count] = random()
            self.count += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.count = np.clip(self.count, 0, TRIALS)

        # Are we at the left of the grid?
        done = bool(self.count == TRIALS - 1)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.count == TRIALS - 1 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.count]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.count, end="")
        print("x", end="")
        print("." * (TRIALS - self.count))

    def close(self):
        pass



# # Test the trained agent
# obs = BadSeedEnv.reset()
# n_steps = 20
# for step in range(n_steps):
#   action, _ = Model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render(mode='console')
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break
