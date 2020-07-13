import numpy as np
import gym
from gym import spaces
from random import random
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

class CustomEnvironment(gym.Env):
    LEFT = 0
    RIGHT = 1

    def __init__(self):
        def __init__(self, grid_size=10):
            super(CustomEnvironment, self).__init__()

            # Size of the 1D-grid
            self.grid_size = grid_size
            # Initialize the agent at the right of the grid
            self.agent_pos = grid_size - 1

            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions, we have two: left and right
            n_actions = 2
            self.action_space = spaces.Discrete(n_actions)
            # The observation will be the coordinate of the agent
            # this can be described both by Discrete and Box space
            self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                                shape=(1,), dtype=np.float32)

    def states(self):
        return dict(type='float', shape=(4,))

    def actions(self):
        return dict(type='int', num_values=2)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    # def max_episode_timesteps(self):
    #     return super().max_episode_timesteps()

    # Optional
    def close(self):
        pass

    def reset(self):
        """
            Important: the observation must be a numpy array
            :return: (np.array)
            """
        # Initialize the agent at the right of the grid
        self.agent_pos = self.grid_size - 1
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def execute(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

            # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

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
        print("." * (self.grid_size - self.agent_pos))


environment = CustomEnvironment()

# Create agent and environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500
# )
agent = Agent.create(agent='a2c', environment=environment, batch_size=10, learning_rate=1e-3)

# Train for 200 episodes
for _ in range(200):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

# Evaluate for 100 episodes
sum_rewards = 0.0
for _ in range(100):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True)
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

print('Mean episode reward:', sum_rewards / 100)

# Close agent and environment
agent.close()
environment.close()