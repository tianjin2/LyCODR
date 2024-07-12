import numpy as np
import time
from collections import deque
import coma.utils.logger as logger
import datetime
import matplotlib
import coma.envs.constant as ct
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Sample(object):
    def __init__(self, max_path_length, min_pool_size, batch_size, agent_num):

        print(max_path_length,min_pool_size)
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._agent_num = agent_num

        self.env = None
        self.agent = None
        self.pool = None


    def initialize(self, env, agent, pool):

        self.env = env
        self.agent = agent
        self.pool = pool

    def set_agent(self, agent):
        self.agent=agent
    def get_agent_num(self):
        return self._agent_num
    def sample(self, t):
        raise NotImplementedError

    def batch_ready(self):

        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples



    def random_batch(self):

        return self.pool.random_batch(self._batch_size)

    def terminate(self):

        self.env.terminate()

    def log_diagnostics(self):

        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sample):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = [0 for i in range(self.get_agent_num())]
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._max_episodes = 10
        self._avg_return = deque([], maxlen=self._max_episodes)

    def sample(self,t):

        if self._current_observation is None:
            self._current_observation = self.env.reset()
        actions, pis, _ = self.agent.get_actions(self._current_observation,t,self._max_path_length, False)

        action = np.array(list(map(list, zip(*actions))))

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return = (np.array(self._path_return) + np.array(reward)).tolist()
        self._total_samples += 1

        self.pool.add_sample(
            agent_num=self.agent.get_agent_num(),
            observation=self._current_observation,
            pi=pis,
            action=actions,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation
        )


        if terminal == [1 for i in range(self.get_agent_num())] or self._path_length >= self._max_path_length:
            self._current_observation = next_observation
            self._path_length = 0
            self._max_path_return = max(self._max_path_return, self._path_length)
            self._last_path_return = self._path_return
            self._avg_return.extend([self._path_return])
            self._path_return = [0 for i in range(self.agent.get_agent_num())]
            self._n_episodes += 0
        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()

        logger.record_tabular('max-path-return', self._max_path_return)


        logger.record_tabular('last-path-return', self._last_path_return)


        logger.record_tabular('avg-path-return', np.mean(self._avg_return))


        logger.record_tabular('episodes', self._n_episodes)


        logger.record_tabular('total-samples', self._total_samples)