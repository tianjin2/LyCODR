import uuid
import copy
from coma.envs import Vehicle_applications, constant

# 交易定义
class Transaction(object):
    # 初始化
    def __init__(self, veh_app_type, data_size, server_index=None, cloud_index=None, sender=None, recipient=None, nonce=None, arrival_timestamp=None):
        self.server_index = server_index
        self.cloud_index = cloud_index
        self.sender = sender
        self.recipient = recipient
        self.nonce = nonce
        self.received_data_size = 0
        self.veh_app_type = veh_app_type
        self.data_size = data_size
        self.is_start = False
        self.is_cross_shard = False
        self.computation_over = 0
        self.uuid = uuid.uuid4()
        self.parent_uuid = None
        self.child_uuid = None
        self.arrival_timestamp = arrival_timestamp
        self.txhash = self.uuid.hex


    def get_uuid(self):
        return self.uuid.hex

    def set_cross_shard(self):
        self.is_cross_shard = True


    def make_sub_transactions(self, server_shard, cloud_shard, sender, recipient, arrival_timestamp):
        sub_transaction = copy.deepcopy(self)
        sub_transaction.server_index = server_shard
        sub_transaction.cloud_index = cloud_shard
        sub_transaction.uuid = uuid.uuid4()
        sub_transaction.sender = sender
        sub_transaction.recipient = recipient
        sub_transaction.child_uuid = None
        sub_transaction.parent_uuid = None
        sub_transaction.arrival_timestamp = arrival_timestamp
        sub_transaction.is_cross_shard = False
        sub_transaction.txhash = sub_transaction.uuid.hex

        return sub_transaction

    def make_child_transaction(self, offload_data_bits):
        new_task = copy.deepcopy(self)
        new_task.uuid = uuid.uuid4()
        new_task.parent_uuid = self.get_uuid()
        self.child_uuid = new_task.get_uuid()
        new_task.data_size = offload_data_bits
        self.data_size -= offload_data_bits
        return new_task


    def get_workload(self):
        return Vehicle_applications.veh_app_info[self.veh_app_type]['workload']


    def get_app_type(self):
        return self.veh_app_type

    def get_data_size(self):
        return self.data_size

    def set_arrival_time(self, arrival_timestamp):
        self.arrival_timestamp = arrival_timestamp


    def get_arrival_time(self):
        return self.arrival_timestamp


    def is_client_index(self):
        return bool(self.server_index)

