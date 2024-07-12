from abc import abstractmethod, ABCMeta

# Define an abstract environment class that cannot be instantiated directly but needs to be inherited by other concrete environment classes and implement their abstract methods
class Environment(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        self.state_dim = 0
        self.action_dim = 0
        self.servers = dict()
        self.clouds = dict()
        self.links = list()
        self.timestamp = 0
        self.silence = True


    @abstractmethod
    def reset(self):
        pass


    @abstractmethod
    def get_status(self):
        pass


    @abstractmethod
    def step(self, actions, t=0, use_beta=True, generate=True):
        pass


    @abstractmethod
    def get_cost(self, used_edge_cpus, used_cloud_cpus, used_edge_bws, used_cloud_bws, before, after, failed_to_offload=0, failed_to_generate=0):
        pass


    def silence_on(self):
        self.silence = True


    def silence_off(self):
        self.silence = False