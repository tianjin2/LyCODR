import logging
import uuid
from coma.envs.transactions_queue import TransactionQueue
from coma.envs.constant import *

logger = logging.getLogger(__name__)

class CloudNode:
    def __init__(self, computational_capability, is_random_transaction_generating=False):
        self.uuid = uuid.uuid4()
        self.links_to_lower = {}  # Communication links to edge servers
        self.links_to_higher = {}
        self.mobility = False  # Whether the node is mobile or not
        self.computational_capability = computational_capability  # Computing power of nodes
        self.transaction_queue = TransactionQueue()  # Transaction queues
        self.cpu_used = 0  # CPU resources currently used

    def __del__(self):
        del self.transaction_queue
        del self

    # Functions that execute transactions and return the percentage of total CPU resources actually used
    def do_transactions(self):
        # If there are tasks in the transaction queue, process them
        if self.transaction_queue.get_length():
            cpu_used_com_cap, _ = self.transaction_queue.served_cloud(self.computational_capability, types=1)
            self.cpu_used = cpu_used_com_cap / (NUM_CLOUD_CLOCK_FREQUENCY * GHZ)
        else:
            self.cpu_used = 0
        return self.cpu_used

    # Check if there is enough space to receive the data planned to be offloaded
    def probed(self, bits_to_be_arrived):
        if self.transaction_queue.get_max() < self.transaction_queue.get_length() + bits_to_be_arrived:
            return (0,True)
        else:
            return (bits_to_be_arrived, False)

    # Process offloaded transactions and return the number of transactions that failed to be offloaded
    def offloaded_transactions(self, transactions, arrival_timestamp):
        failed_to_offload = 0
        for transaction_id, transaction_ob in transactions.items():
            transaction_ob.server_index = transaction_ob.server_index
            transaction_ob.cloud_index = self.get_uuid()
            transaction_ob.set_arrival_time(arrival_timestamp)
            # Attempt to add transaction to queue, if fails increase failure counter
            failed_to_offload += (not self.transaction_queue.arrived_cloud(transaction_ob, arrival_timestamp))
        return failed_to_offload  # 返回未能卸载的交易数量

    # Get the length of the transaction queue
    def get_transaction_queue_length(self, scale=1):
        return self.transaction_queue.get_length(scale=scale)

    def sample_channel_rate(self, linked_id):
        # If uplink, return uplink channel rate
        if linked_id in self.links_to_higher.keys():
            return self.links_to_higher[linked_id]['channel'].get_rate()
        # If downlink, return downlink channel rate
        elif linked_id in self.links_to_lower.keys():
            return self.links_to_lower[linked_id]['channel'].get_rate(False)

    # Get the unique identifier of the cloud server node
    def get_uuid(self):
        return self.uuid.hex

    def _get_obs(self, time, estimate_interval=100, involve_capability=False, scale=1):
        # Returns the average arrival rate of transactions, the last arrival rate, the CPU requirements of the queue and the percentage of CPU used
        return [
            self.transaction_queue.mean_arrival(time, estimate_interval, scale=scale),
            self.transaction_queue.last_arrival(time, scale=scale),
            self.transaction_queue.get_cpu_needs(scale=GHZ*NUM_CLOUD_CLOCK_FREQUENCY),
            self.cpu_used / 54
        ]