import gym  # 导入gym库，用于创建和管理各种强化学习环境
import gym.wrappers  # 导入 gym 的 wrappers 模块，提供环境的额外层封装功能
import gym.envs  # 导入 gym 的 envs 模块，提供各种内置的环境
import gym.spaces  # 导入 gym 的 spaces 模块，提供状态空间和动作空间的定义
import traceback  # 导入 traceback 模块，用于输出错误和异常信息
import logging  # 导入 logging 模块，用于实现日志记录
import coma.utils.logger as logger
from coma.utils.serializable import Serializable
import collections
import numpy as np



class GymEnv(gym.Env, Serializable):
    def __init__(self, env_name, agent_num,record_video=False, video_schedule=None, log_dir=None, record_log=False,
                 force_reset=True, obs_alpha=0.001, reward_alpha=0.001):

        if log_dir is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")

        Serializable.quick_init(self, locals())
        self.monitoring = False

        env = gym.envs.make(env_name)

        env = env.env
        self.env = env
        self.env_id = env.spec.id
        self._observation_space = env.observation_space
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = env.action_space
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(6 * agent_num)
        self._obs_var = np.ones(6 * agent_num)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.


    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon


    def reset(self):

        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True

        return self.env.reset()


    def step(self, action):

        next_obs, reward, terminal, info = self.env.step(action)
        return Step(next_obs, reward, terminal, info)



    def render(self, mode='human', close=False):

        return self.env._render(mode, close)



    def terminate(self):

        if self.monitoring:
            self.env._close()

            if self._log_dir is not None:
                print("""
                ***************************
            
                Training finished! You can upload results to OpenAI Gym by running the following command:
            
                python scripts/submit_gym.py %s
            
                ***************************
                            """ % self._log_dir)

    def _update_obs_estimate(self, obs):
        flat_obs = np.asarray(obs).flatten()

        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var + self._reward_alpha * np.square(reward - self._reward_mean)

    def log_diagnostics(self, paths):
        for path in paths:
            observations = path['observations']
            for observation in observations:
                self._update_obs_estimate(observation)
            rewards = path['rewards']
            for reward in rewards:
                self._update_reward_estimate(reward)
        print("Obs mean:", self._obs_mean)
        print("Obs std:", np.sqrt(self._obs_var))
        print("Reward mean:", self._reward_mean)
        print("Reward std:", np.sqrt(self._reward_var))

Step = collections.namedtuple("Step", ["observation", "reward", "terminal", "info"])


class GymEnvDelayed(gym.Env):
    def __init__(self, env_name, record_video=False, video_schedule=None, log_dir=None, record_log=False,
                 force_reset=True, delay = 0):
        if log_dir is None:
            logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
        env = gym.envs.make(env_name)
        env = env.env
        self.env = env
        self.env_id = env.spec.id
        self.time = 0
        self.delay = delay
        self.delay_reward = 0
        self._observation_space = env.observation_spac
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = env.action_space
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset



