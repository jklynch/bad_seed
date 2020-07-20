import numpy as np
import gym
from gym import spaces
from random import random
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from heapq import nlargest


def stdDeviaiton(array):
    cleanedUp = np.array([])
    for elem in array:
        if elem != 0:
            cleanedUp = np.append(cleanedUp, elem)
    return np.std(cleanedUp)

class CustomEnvironment(Environment):
    # LEFT = 0
    # RIGHT = 1
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
            self.TRIALS = 20
            self.SAMPLES = 5
            self.GRID = []
            self.minSampling = {}
            self.stdDev = {}
            # self.stdDevSim = {}
            self.sum = 0
            # self.simulation = [[0, 0, 0, 0, 0, 0, 7, 2, 0, 0], [0, 3, 0, 0, 0, 3, 0, 0, 0, 0],[0, 0, 2, 9, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 8, 0]]


            for i in range(self.SAMPLES):
                col = []
                for j in range(self.TRIALS):
                    col.append(0)
                self.GRID.append(col)

            for i in range(self.SAMPLES):
                self.minSampling[i] = 0

            for i in range(self.SAMPLES):
                self.stdDev[i] = 0

            # for i in range(self.SAMPLES):
            #     self.stdDevSim[i] = 0
            # for i in range(self.SAMPLES):
                # print(i)
                # print(self.simulation[i])
                # print(stdDeviaiton(array=[0, 0, 0, 0, 0, 0, 1, 0, 8, 0]))
                # self.stdDevSim[i] = stdDeviaiton(array=self.simulation[i])
                # print(self.stdDevSim)
            # print(nlargest(3, self.stdDevSim, key=self.stdDevSim.get))


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

    def execute(self, actions):
        maxStdDev = []
        reward = 0
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
            for i in range(self.SAMPLES):
                self.stdDev[i] = stdDeviaiton(array=self.GRID[i])
            maxStdDev = nlargest(3, self.stdDev, key=self.stdDev.get)
            if actions in maxStdDev:
                reward += 1
                print(reward)
            # print(maxStdDev, actions)
            # if self.agent_pos <= self.TRIALS:
            print(self.agent_pos)
            self.GRID[actions][self.agent_pos] = random()
            self.minSampling[actions] += 1
            self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(actions))
            # Account for the boundaries of the grid
        # self.agent_pos = np.clip(self.agent_pos, 0, self.TRIALS)

        # Are we at the right of the grid?
        done = bool(self.agent_pos + 1 >= self.TRIALS - 1)

        # if actions in maxStdDev:
        #     reward += 1
        #     print(reward)
        self.agent_pos += 1
        print(self.agent_pos, self.TRIALS)
        if self.agent_pos + 1 == self.TRIALS - 1:
            reward += 5
        elif self.agent_pos + 2 == self.TRIALS - 1:
            reward += 4
        elif self.agent_pos + 3 == self.TRIALS - 1:
            reward += 3
        elif self.agent_pos + 4 == self.TRIALS - 1:
            reward += 2
        else:
            reward += 1

        if done:
            print("reward", reward)
            self.sum += 1
            if self.sum > 20:
                CustomEnvironment.sum += reward
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