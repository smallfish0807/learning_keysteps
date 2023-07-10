import numpy as np

import gym
from gym import spaces


class ToyMDP(gym.Env):
    def __init__(self, epsilon):
        # 0 means "no attack", 1 means "attack"
        self.action_space = spaces.Discrete(2)

        self.state_num = 15
        low = np.zeros(self.state_num, dtype=int)
        high = np.ones(self.state_num, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.uint8)

        # We swap state s_5 and s_6 in the toy mdp for simple implementation of
        # reward.
        self.reward_list = [
            0, 0, 0, 0, 0, 0, 0, 1, -1, 0.99, -1, 1, -10, 0, -1
        ]
        self.epsilon = epsilon

        self.reset()

    def step(self, action):
        assert action == 0 or action == 1

        if self.done:
            return self._onehot(self.state_id), 0, self.done, {}

        self.state_id = self.state_id * 2 + action + 1
        reward = -self.reward_list[self.state_id] - action * self.epsilon
        done = (self.state_id >= 7)

        return self._onehot(self.state_id), reward, done, {}

    def reset(self):
        self.state_id = 0
        self.done = False
        return self._onehot(self.state_id)

    def render(self):
        pass

    def _onehot(self, n):
        return np.array([0] * n + [1] + [0] * (self.state_num - n - 1))


class ToyMDP_1(ToyMDP):
    def __init__(self):
        super().__init__(10)


class ToyMDP_2(ToyMDP):
    def __init__(self):
        super().__init__(1)


class ToyMDP_3(ToyMDP):
    def __init__(self):
        super().__init__(0.1)
