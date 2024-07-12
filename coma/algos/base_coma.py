import abc
from coma.misc import tf_utils
import tensorflow as tf
import gtimer as gt
import coma.utils.logger as logger
from coma.core.serializable import deep_clone

class RLAlgorithm():
    def __init__(self,
                 sampler,
                 n_epochs=1000.0,
                 n_train_repeat=0,
                 n_initial_exploration_steps=10000,
                 epoch_length=15000,
                 eval_n_episodes=10,
                 eval_deterministic=True,
                 eval_render=True,
                 ):
        """
       参数：
           n_epochs（`int`）： The number of epochs for running training.
           n_train_repeat （`int`）： The number of repetitions of the training in a single time step.
           n_initial_exploration_steps： Use the number of steps at the start of an operation extracted from a separate exploration policy.
           epoch_length (`int`)： time length。
           eval_n_episodes (`int`)： the episodes of the evaluation。
           eval_deterministic (`int`)： Whether the policy run in deterministic mode when evaluating the policy.
           eval_render (`int`)： Whether or not to render the evaluation environment.
       """

        # Initialisation function to set various parameters for training
        self.sampler = sampler  # 采样器，用于生成训练数据
        self._n_epochs = int(n_epochs)  # 训练的周期数
        self._n_train_repeat = n_train_repeat # 每个时间步重复训练的次数
        self._epoch_length = epoch_length # 每个周期的长度
        self._n_initial_exploration_steps = n_initial_exploration_steps  # 初始化探索步骤数

        self._eval_n_episodes = eval_n_episodes  # 评估周期数 （后续是否删掉）
        self._eval_deterministic = eval_deterministic  # 是否使用确定性策略进行评估
        self._eval_render = eval_render  # 是否在评估时渲染环境

        self._sess = tf_utils.get_default_session()

        self._env = None  # 环境，将在后面设置
        self._agent = None  # 策略，将在后面设置
        self._pool = None  # 数据池，用于存储样本，将在后面设置

    def _train(self, env, agent, pool, initial_exploration_done=False):
        self.sampler.initialize(env, agent, pool)
        self._init_training(env=env, agent=agent, pool=pool,epoch=0)

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')  # Set the root name of gtimer
            gt.reset()  # Reset gtimer
            gt.set_def_unique(False)  # Set the default uniqueness of the gtimer
            for epoch in gt.timed_for(range(self._n_epochs +1), save_itrs=True):
                logger.push_prefix('Epoch #%d ' % epoch)  # Add log prefix
                for t in range(self._epoch_length):
                    if not initial_exploration_done:
                        if self._epoch_length * epoch >= self._n_initial_exploration_steps:
                            self.sampler.set_agent(agent)
                            initial_exploration_done = True
                    self.sampler.sample(t + epoch * self._epoch_length)  # 进行采样
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')  # Record sampling time

                    for i in range(self._n_train_repeat):   # Repetition training
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch(),
                        )
                    gt.stamp('train')  # Recording training time


                # Save training parameters and logs
                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)

                self.sampler.log_diagnostics()  # Record sampler diagnostic information

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
                gt.stamp('eval')  # Record the time of the evaluation

            self.sampler.terminate()  # Terminate the sampler


    @abc.abstractmethod
    def _init_training(self, env, agent, pool, epoch):
        #  Methods for initialising training. Set up the environment, policy, and sample pool before starting training.
        self._env = env
        self._agent = agent
        self._pool = pool

        # If an evaluation is required, clone an environment for the evaluation
        if self._eval_n_episodes>0:
            self._eval_env = deep_clone(env)
        if epoch > 0:
            self._agent = deep_clone(agent)



    @abc.abstractmethod
    def get_snapshot(self, epoch):
        # Used to get a snapshot of a moment in the training process
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        # Abstract methods used to perform training
        raise NotImplementedError

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent

    @property
    def pool(self):
        return self._pool


