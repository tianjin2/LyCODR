import abc

class ReplayBuffer(object, metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def add_sample(self, agent_num, observation, pi, action, reward, next_observation, terminal, **kwargs):

        pass

    @abc.abstractmethod
    def terminate_episode(self):

        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):

        pass

    def add_path(self, path):

        for agent_num, pi, obs, action, reward, next_obs, terminal, agent_info, env_info in zip(
                path["agent_num"],
                path["observations"],
                path["pi"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
        ):
            self.add_sample(
                agent_num,
                obs,
                pi,
                action,
                reward,
                terminal,
                next_obs,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):

        pass