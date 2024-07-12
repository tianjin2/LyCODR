# Define the cache, TransactionBuffer is used to store the task data, ReplayBuffer is used in the reinforcement learning


from flatbuffers.builder import np

# 定义TaskBuffer类
class TransactionBuffer:
    def __init__(self, max_size=100):
        # Initialise the list for storage
        self.storage = list()
        self.max_size = max_size

    # adding data
    def add(self, data):
        self.storage.append(data)
        # Control the size
        if len(self.storage)>self.max_size:
            self.storage = self.storage[1:]
        else:
            pass

    # Getting data
    def get_buffer(self):
        return self.storage[::-1]

    # Get the last object stored
    def get_last_obs(self):
        if self.storage:
            return self.storage[-1]
        else:
            return None

# define ReplayBuffer class
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        # Initialise pointer to 0 for cyclic data storage
        self.ptr = 0

    # adding data
    def add(self,data):
        if len(self.storage)==self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr+1) % self.max_size
        else:
            self.storage.append(data)

    # Extract a batch from the buffer
    def sample(self, batch_size):
        # Random batch of random indexes
        index = np.random.randint(0, len(self.storage), size=batch_size)

        x, y, u, r, d = [], [], [], [], []

        # Iterate over the selected index and get the data
        for i in index:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        # Return the extracted samples in the form of an array, adjusted to the shape of a line
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1,1), np.array(d).reshape(-1,1)



