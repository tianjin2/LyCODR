import numpy as np  # 导入numpy库，用于科学计算

def softmax_1d(array):
    array = np.exp(array - np.max(array))   # 数组中的每个元素减去数组中的最大值，然后计算e的指数
    return array / array.sum()  # 返回正规化的数组，其元素值的和为1

class Momory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []  # 存储动作的对数概率
        self.rewards = []
        self.is_terminals = []  # 存储是否达到终止状态的标记

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]