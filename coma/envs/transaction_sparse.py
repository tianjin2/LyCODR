

class RewardFunction:

    def __ceil__(self, state, action, next_state):
        raise NotImplementedError  # 抛出错误，需要子类重写这个方法

class Transaction:
    def __init__(self, measure, reward_function):
        self.measure = measure
        self.reward_function = reward_function  #


class Measure:
    def __ceil__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """
                compute utilities of each policy
                Args:
                    states: (n_actors, d_state)
                    actions: (n_actors, d_action)
                    next_state_means: (n_actors, ensemble_size, d_state)
                    next_state_vars: (n_actors, ensemble_size, d_state)
                Returns:
                    utility: (n_actors)
        """
        raise NotImplementedError  #
