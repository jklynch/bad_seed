import numpy as np
import gym
from gym import spaces
from random import random
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

class CustomEnvironment(Environment):
    LEFT = 0
    RIGHT = 1
    sum = 0
    # SAMPLES = 5
    # TRIALS = 10
    # GRID = []


    # def __init__(self):
    def __init__(self, grid_size=10):
            super().__init__()

            # Size of the 1D-grid
            self.grid_size = grid_size
            # Initialize the agent at the right of the grid
            self.agent_pos = 0
            self._max_episode_timesteps = 500
            self.TRIALS = 100
            self.SAMPLES = 10
            self.GRID = []
            self.minSampling = {}
            self.stdDev = {}
            self.sum = 0

            for i in range(self.SAMPLES):
                col = []
                for j in range(self.TRIALS):
                    col.append(0)
                self.GRID.append(col)

            for i in range(self.SAMPLES):
                self.minSampling[i] = 0

            for i in range(self.SAMPLES):
                self.stdDev[i] = 0

            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions, we have two: left and right
            n_actions = self.SAMPLES
            self.action_space = spaces.Discrete(n_actions)
            # The observation will be the coordinate of the agent
            # this can be described both by Discrete and Box space
            self.observation_space = spaces.Box(low=0, high=self.SAMPLES,
                                                shape=(self.SAMPLES, self.TRIALS), dtype=np.float32)

    def states(self):
        return dict(type='float', shape=(1))

    def actions(self):
        return dict(type='int', num_values=self.SAMPLES)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()
    #
    # # Optional
    # def close(self):
    #     pass

    def reset(self):
        """
            Important: the observation must be a numpy array
            :return: (np.array)
            """
        for i in range(self.SAMPLES):
            for j in range(self.TRIALS):
                self.GRID[i][j] = 0

        for i in range(self.SAMPLES):
            self.minSampling[i] = 0
        # Initialize the agent at the right of the grid
        self.agent_pos = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def stdDeviaiton(self, array):
        cleanedUp = np.array([])
        for elem in array:
            if elem != 0:
                np.append(cleanedUp, elem)
        return np.std(cleanedUp)

    def execute(self, actions):
        if self.agent_pos < self.SAMPLES:
            for i in range(self.SAMPLES):
                self.GRID[i][self.agent_pos] = random()
                self.minSampling[i] += 1
                self.agent_pos += 1
        elif self.agent_pos < 2 * self.SAMPLES:
            for i in range(self.SAMPLES):
                self.GRID[i][self.agent_pos] = random()
                self.minSampling[i] += 1
                self.agent_pos += 1

        elif (actions >= 0 and actions < self.SAMPLES):
            if self.agent_pos <= self.TRIALS:
                # print(self.agent_pos)
                self.GRID[actions][self.agent_pos] = random()
                self.minSampling[actions] += 1
                # self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(actions))
            # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.TRIALS)

        # Are we at the right of the grid?
        done = bool(self.agent_pos >= self.TRIALS - 1)
        if done:
            print(self.minSampling)
            # print("blah", self.GRID[0][0])
            # print("blah", self.GRID[1][0])
            # print("blah", self.GRID[2][0])
            # print("blah", self.GRID[3][0])
            # print("blah", self.GRID[4][0])
        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 0
        # reward = 1 if self.agent_pos == self.TRIALS - 1 else 0
        # if done:
        # for i in range(self.SAMPLES):
        #     if self.minSampling[i] <= 2 and done:
        #         reward += 1

        if self.agent_pos == self.TRIALS - 1:
            reward += 1

        if done:
            print("woot")
            self.sum += 1
            if self.sum > 20:
                CustomEnvironment.sum += reward
                # print(self.sum)
                # print(reward)
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        # print(self.SAMPLES)
        # print(self.agent_pos)
        if self.agent_pos >= 2 * self.SAMPLES:
            self.agent_pos += 1
        returning = np.array([self.agent_pos]).astype(np.float32), reward, done
        # print(done)
        return returning


environment = Environment.create(
    environment=CustomEnvironment, max_episode_timesteps=500
)

# Create agent and environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500
# )
agent = Agent.create(agent='a2c', environment=environment, batch_size=10, learning_rate=1e-3)

# Train for 200 episodes
for _ in range(20):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        # print(actions)
        # print(states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

# Evaluate for 100 episodes
sum_rewards = 0.0
for _ in range(10):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True)
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

# print('Mean episode reward:', sum_rewards / 100)
print(CustomEnvironment.sum)

# Close agent and environment
agent.close()
environment.close()