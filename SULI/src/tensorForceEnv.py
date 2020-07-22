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
    extraCounter = 3
    firstCount = 0
    secondCount = 0
    thirdCount = 0
    # SAMPLES = 5
    # TRIALS = 10
    # GRID = []


    # def __init__(self):
    def __init__(self, grid_size=10):
            super().__init__()

            self.startingPoint = 3

            # Size of the 1D-grid
            self.grid_size = grid_size
            # Initialize the agent at the right of the grid
            self.agent_pos = self.startingPoint
            self._max_episode_timesteps = 500
            self.TRIALS = 100
            self.SAMPLES = 5
            self.GRID = []
            self.minSampling = {}
            self.stdDev = {}
            # self.stdDevSim = {}
            self.sum = 0
            self.reward = 0
            # self.extraCounter = self.startingPoint
            # self.simulation = [[0, 0, 0, 0, 0, 0, 7, 2, 0, 0], [0, 3, 0, 0, 0, 3, 0, 0, 0, 0],[0, 0, 2, 9, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 8, 0]]


            for i in range(self.SAMPLES):
                col = []
                for j in range(self.TRIALS):
                    if j < self.startingPoint:
                        col.append(random())
                    else:
                        col.append(0)
                self.GRID.append(col)

            for i in range(self.SAMPLES):
                self.minSampling[i] = 0

            for i in range(self.SAMPLES):
                self.stdDev[i] = self.startingPoint

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
        # self.extraCounter = self.startingPoint
        CustomEnvironment.extraCounter = self.startingPoint
        self.reward = 0
        self.agent_pos = self.startingPoint
        for i in range(self.SAMPLES):
            for j in range(self.TRIALS):
                if j < self.startingPoint:
                    self.GRID[i][j] = random()
                else:
                    self.GRID[i][j] = 0

        for i in range(self.SAMPLES):
            self.minSampling[i] = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def execute(self, actions):
        # self.extraCounter += 1
        CustomEnvironment.extraCounter += 1
        maxStdDev = []
        reward = 0
        if (actions >= 0 and actions < self.SAMPLES):
            for i in range(self.SAMPLES):
                self.stdDev[i] = stdDeviaiton(array=self.GRID[i])
            maxStdDev = nlargest(3, self.stdDev, key=self.stdDev.get)
            print(actions, maxStdDev)
            if actions == maxStdDev[0]:
                CustomEnvironment.firstCount += 1
                self.reward += 1
            if actions == maxStdDev[1]:
                CustomEnvironment.secondCount += 1
                self.reward += 1
            if actions == maxStdDev[2]:
                CustomEnvironment.thirdCount += 1
                self.reward += 1
            # print(maxStdDev, actions)
            # if self.agent_pos <= self.TRIALS:
            self.GRID[actions][self.agent_pos] = random()
            self.minSampling[actions] += 1
            self.agent_pos += 1
            print(self.reward)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(actions))
            # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.TRIALS)

        # Are we at the right of the grid?
        done = bool(self.agent_pos == self.TRIALS)

        if done:
            reward = self.reward
            # reward += 1
            self.sum += 1
            if self.sum > 0:
                CustomEnvironment.sum += reward
            print(self.minSampling)
        returning = np.array([self.agent_pos]).astype(np.float32), reward, done
        return returning

def runEnv():
    environment = Environment.create(
        environment=CustomEnvironment, max_episode_timesteps=500
    )
    agent = Agent.create(agent='a2c', environment=environment, batch_size=10, learning_rate=1e-3)

    # Train for 200 episodes
    # for _ in range(2):
    #     states = environment.reset()
    #     terminal = False
    #     while CustomEnvironment.extraCounter != 100:
    #         actions = agent.act(states=states)
    #         # print(actions)
    #         # print(states)
    #         states, terminal, reward = environment.execute(actions=actions)
    #         agent.observe(terminal=terminal, reward=reward)

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(1):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while CustomEnvironment.extraCounter != 100:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward

    # print('Mean episode reward:', sum_rewards / 100)
    print(CustomEnvironment.firstCount)
    print(CustomEnvironment.secondCount)
    print(CustomEnvironment.thirdCount)
    print(CustomEnvironment.sum)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == "__main__":
    runEnv()
