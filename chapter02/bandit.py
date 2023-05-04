import gym
import gym.spaces
import gym.utils.seeding
import gym.envs.registration

import numpy as np
import matplotlib.pyplot as plt


class ArmedBanditTestbedEnv(gym.Env):
    """
    Armed Bandit Testbed per Example 2.2 (stationary=True) or Exercise 2.5 (stationary=False) 
    #q_star (true values of the expected rewards of each arm) 
    """
    def __init__(self, n=10, stationary=True):
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(n)
        self.stationary = stationary
        self.np_random = None
        self.seed()

        self.q_star = None 
        self.a_star = None
        self.reset()

    def step(self, arm):
        """
        The step method takes an action (an integer representing which arm the gambler chooses) 
        and returns the reward for that action, 
        whether the episode is done, and some additional information (the best arm).
        """
        assert self.action_space.contains(arm)

        if self.stationary is False:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.action_space.n)
            self.q_star += q_drift
        reward = self.np_random.normal(loc=self.q_star[arm], scale=1.0)
        return 0, reward, False, {'arm_star': np.argmax(self.q_star)}

    def reset(self):
        if self.stationary is True:
            self.q_star = self.np_random.normal(loc=0.0, scale=1.0, size=self.action_space.n)
        else:
            self.q_star = np.zeros(shape=[self.action_space.n])

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [np.random.normal(loc=self.q_star[arm], scale=1.0, size=1_000)]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel('Bandit Arm')
        plt.ylabel('Reward Distribution')
        plt.show()


gym.envs.registration.register(
    id='ArmedBanditTestbed-v0',
    entry_point=lambda n, stationary: ArmedBanditTestbedEnv(n, stationary),
    max_episode_steps=1_000,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={'n': 10, 'stationary': True}
)


def run_episode(bandit, gambler, seed=None):
    """
    Execute a single episode of the Armed Bandit Testbed environment.
    
    executes a single episode of the environment 
    and returns the rewards, whether the gambler chose the best arm, 
    and the number of steps taken. 
    The function takes two inputs: 
    1. the environment 
    2. and the gambler 
    (an object with a method called "arm" that chooses which arm to pull 
    and a method called "update" that updates its estimate of 
    the expected reward of the chosen arm based on the actual reward received).
    """
    rewards, corrects, steps = [], [], 0

    bandit.seed(seed)
    bandit.reset()

    gambler.seed(seed)
    gambler.reset()

    done = False
    while not done:
        arm = gambler.arm()
        state, reward, done, info = bandit.step(arm)
        gambler.update(arm, reward)

        rewards += [reward]
        corrects += [1 if arm == info['arm_star'] else 0]
        steps += 1

    rewards = np.array(rewards, dtype= float)
    corrects = np.array(corrects, dtype=int)
    return rewards, corrects, steps
