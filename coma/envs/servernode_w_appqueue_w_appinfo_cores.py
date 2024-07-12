import secrets
import string
import logging
import uuid
import numpy as np
import random
from coma.envs.constant import *
from coma.envs import Vehicle_applications
from coma.envs.transaction import Transaction
from coma.envs.transactions_queue import TransactionQueue

# 设置日志记录器
logger = logging.getLogger(__name__)


class ServerNode:
    def __init__(self, uuid_node, computational_capability, bws, shard_list=None, is_random_generating=False):
        self.shard_list = shard_list
        self.uuid = uuid_node
        self.links_to_lower = {}  # Linking information to vehicles
        self.links_to_higher = {}  # Connection information to cloud nodes

        self.computational_capability = computational_capability
        self.bws = bws
        self.number_of_veh_applications = 0  #  Number of initialised applications
        self.queue_list = {}

        self.is_random_transaction_generating = is_random_generating
        self.cpu_used = {}
        self.bw_used = {}
        self.bw_used_offload = {}
        self.account = []

    def __del__(self):
        iter = list(self.queue_list.keys())
        for veh_app_type in iter:
            del self.queue_list[veh_app_type]
        del self


    def make_veh_application_queues(self, *veh_application_types):
        # Iterate over all applications and create transaction queues for each application
        for veh_application_type in veh_application_types:
            self.queue_list[veh_application_type] = TransactionQueue(veh_application_type)
            self.number_of_veh_applications += 1
            self.cpu_used[veh_application_type] = 0
            self.bw_used[veh_application_type] = 0
            self.bw_used_offload[veh_application_type]=0
        return

    def set_account(self, account_num):
        for _ in range(account_num):
            self.account.append(self.generation_address())

    def get_account(self):
        return self.account

    def generation_address(self):
        random_address = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
        return random_address

    def get_cpu(self):
        return self.computational_capability

    def get_bw(self):
        return self.bws


    def do_transactions(self, resource):
        # Get all app types in the queue list
        veh_type_list = list(self.queue_list.keys())
        res_allocs = dict()
        flags = []
        for veh_app_type in veh_type_list:
            my_transaction_queue = self.queue_list[veh_app_type]
            if my_transaction_queue.get_length() or resource == [0.,0.]:
                res_allocs[veh_app_type], resource, _ = my_transaction_queue.served(resource, types=1)
            else:
                res_allocs[veh_app_type] = (0, 0)
        self.cpu_used = dict(zip(res_allocs.keys(), [x[0] for x in res_allocs.values()]))
        self.bw_used = dict(zip(res_allocs.keys(), [x[1] for x in res_allocs.values()]))

        return sum(self.cpu_used.values()), sum(self.bw_used.values())


    def _probe(self, bits_to_be_arrived,id_to_offload):

        node_to_offload = self.links_to_higher[id_to_offload]['node']
        failed = {}

        for veh_app_type, bits in bits_to_be_arrived.items():
            if (veh_app_type in self.queue_list.keys()):
                bits_to_be_arrived[veh_app_type], failed[veh_app_type] = node_to_offload.probed(bits)
        return bits_to_be_arrived, failed

    def probed(self, veh_app_type, bits_to_be_arrived):

        if self.queue_list[veh_app_type]:
            # 检查长度
            if self.queue_list[veh_app_type].get_max() < self.queue_list[veh_app_type].get_length() + bits_to_be_arrived:
                return (0, True)
            else:
                return (bits_to_be_arrived, False)
        return (0,True)


    def sample_channel_rate(self, linked_id):

        if linked_id in self.links_to_higher.keys():
            return self.links_to_higher[linked_id]['channel'].get_rate()

        elif linked_id in self.links_to_lower.keys():
            return self.links_to_lower[linked_id]['channel'].get_rate(False)


    def get_queue_lengths(self, scale=1):
        lengths = np.zeros(len(self.queue_list))
        for veh_app_type, queue in self.queue_list.items():
            lengths[veh_app_type-1] = queue.get_length(scale)
        return lengths

    def offload_transactions(self, beta, id_to_offload):
        channel_rate = self.sample_channel_rate(id_to_offload)
        veh_app_list = list(self.queue_list.keys())

        veh_length = self.get_queue_lengths()
        lengths = sum(self.get_queue_lengths())

        bw_alloc = min(lengths, beta  * channel_rate)

        tx_allocs = {}

        for veh_app_type in veh_app_list:
            if veh_length[veh_app_type-1] <= bw_alloc:
                tx_allocs[veh_app_type] = veh_length[veh_app_type-1]
                bw_alloc -= veh_length[veh_app_type-1]
            else:
                tx_allocs[veh_app_type] = bw_alloc
                bw_alloc = 0

        tx_allocs, failed = self._probe(tx_allocs, id_to_offload)

        transaction_to_be_offloaded = {}
        for veh_app_type in veh_app_list:
            if tx_allocs[veh_app_type] == 0 or (veh_app_type not in list(self.queue_list.keys())):
                pass
            else:
                my_transaction_queue = self.queue_list[veh_app_type]
                if my_transaction_queue.get_length():

                    tx_allocs[veh_app_type], new_to_be_offloaded = my_transaction_queue.served_offload(tx_allocs[veh_app_type],types=0)
                    transaction_to_be_offloaded.update(new_to_be_offloaded)
                else:
                    tx_allocs[veh_app_type] = 0  # 如果队列为空，则设置分配为0
            self.bw_used[veh_app_type] += tx_allocs[veh_app_type] / self.sample_channel_rate(self.get_higher_node_ids()[0])


        return sum(tx_allocs.values()), transaction_to_be_offloaded, failed

    # It should actually be a vehicle unloading transaction, so for simulation purposes, the transaction is generated here directly
    def random_transaction_generation(self, transaction_rate, arrival_timestamp, *veh_app_types):
        veh_app_type_pop = Vehicle_applications.veh_app_type_pop()  # Get the popularity of the app
        this_veh_app_type_list = list(self.queue_list.keys()) # Get the type of application currently in the queue
        random_id = uuid.uuid4()
        arrival_size = np.zeros(len(veh_app_types))
        failed_to_generate = 0
        for veh_app_type, population in veh_app_type_pop:
            if veh_app_type in this_veh_app_type_list:
                # Generate data sizes based on Poisson distribution and prevalence
                data_size = np.random.poisson(transaction_rate*population) * Vehicle_applications.arrival_bits(veh_app_type)
                if data_size > 0:
                    sender, recipient = self.generation_sender_recipient()

                    transaction = Transaction(veh_app_type, data_size, server_index=self.get_uuid(), cloud_index=random_id.hex,sender=sender, recipient=recipient, arrival_timestamp=arrival_timestamp)
                    failed_to_generate += (not self.queue_list[veh_app_type].arrived(transaction, arrival_timestamp))
                    arrival_size[veh_app_type-1] = data_size
        return arrival_size, failed_to_generate

    def generation_sender_recipient(self):
        sender = random.choice(self.account)
        recipient = random.choice(self.account)
        while(sender == recipient):
            recipient = random.choice(self.account)
        return sender, recipient

    def get_higher_node_ids(self):
        return list(self.links_to_higher.keys())


    def get_queue_list(self):
        return self.queue_list.items()


    def get_veh_applications(self):
        return list(self.queue_list.keys())

    def get_uuid(self):
        return self.uuid.hex

    # Obtain observation information of the node, including estimated transaction arrival information, actual transaction arrival information, queue length, CPU resources used, etc.
    def _get_obs(self, time, estimate_interval=100, scale=1):
        queue_estimated_arrivals = np.zeros(1)
        queue_arrivals = np.zeros(1)
        queue_lengths = np.zeros(1)
        bw_used = np.zeros(1)
        cpu_used = np.zeros(1)
        queue_estimated_arrivals[0] = sum([que.mean_arrival(time, estimate_interval, scale=scale) for _, que in self.queue_list.items()])/10
        queue_arrivals[0] = sum([que.last_arrival(time, scale=scale) for _, que in self.queue_list.items()])/10
        queue_lengths[0] = sum([que.get_length(scale=scale) for _, que in self.queue_list.items()])/10

        bw_used[0] = sum(self.bw_used.values()) / 200
        cpu_used[0] = sum(self.cpu_used.values()) /200

        return list(queue_estimated_arrivals) + list(queue_arrivals) + list(queue_lengths) + list(cpu_used) + list(bw_used)

