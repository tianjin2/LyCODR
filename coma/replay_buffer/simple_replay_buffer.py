import numpy as np

from .replay_buffer import ReplayBuffer
from coma.utils.serializable import Serializable

class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, agent_num, env, max_replay_buffer_size):
        super(SimpleReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        self._agent_num = agent_num

        self._env_spec = env.spec

        self._observation_dim = 6

        self._action_dim = 3
        max_replay_buffer_size = int(max_replay_buffer_size)

        self._max_buffer_size = max_replay_buffer_size

        self._pis = np.zeros((max_replay_buffer_size, self._agent_num,self._action_dim))

        self._observations = np.zeros((max_replay_buffer_size,self._agent_num, self._observation_dim))

        self._next_obs = np.zeros((max_replay_buffer_size,self._agent_num, self._observation_dim))

        self._actions = np.zeros((max_replay_buffer_size, self._agent_num,self._action_dim))

        self._rewards = np.zeros((max_replay_buffer_size, self._agent_num))

        self._terminals = np.zeros((max_replay_buffer_size, agent_num), dtype='uint8')

        self._top = 0

        self._size = 0


    def add_sample(self, agent_num, observation, pi, action, reward, next_observation, terminal, **kwargs):
        self._agent_num = agent_num
        self._observations[self._top] = observation
        self._actions[self._top] = action
        pi = np.array(pi)

        self._pis[self._top] = pi
        self._next_obs[self._top] = next_observation
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._advance()

    def string(self):
        return self._max_buffer_size


    def terminate_episode(self):
        pass


    def _advance(self):
        self._top = int((self._top + 1) % self._max_buffer_size)
        if self._size < self._max_buffer_size:
            self._size += 1

    def get_reward(self):
        return self._rewards
    def get_lengt(self):
        return self._observations


    def random_batch(self, batch_size):

        indices = np.random.randint(0, self._size, batch_size)
        if not np.any(indices == 1):
            indices[batch_size-1] = self._size -1

        return dict(
            agent_num=self._agent_num,
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            pis=self._pis[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )


    @property
    def size(self):
        return self._size


    def __getstate__(self):
        d = super(SimpleReplayBuffer, self).__getstate__()
        d.update(dict(
            ag=self._agent_num,
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            p=self._pis.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size
        ))
        return d


    def __setstate__(self, d):
        super(SimpleReplayBuffer, self).__setstate__(d)

        self._agent_num = d['ag']
        self._observations = np.fromstring(d['o']).reshape(self._max_buffer_size,-1)
        self._next_obs = np.fromstring(d['no']).reshape(self._max_buffer_size, -1)
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._pis = np.fromstring(d['p']).reshape(self._max_buffer_size, self._agent_num, -1)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8).reshape(self._max_buffer_size,-1)
        self._top = d['top']
        self._size = d['size']