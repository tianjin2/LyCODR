
import logging
import collections
import numpy as np

from coma.envs.transaction import *
from coma.envs.buffers import TransactionBuffer
from coma.envs import constant as ct


logger = logging.getLogger(__name__)

class TransactionQueue(object):
    def __init__(self, veh_app_type=None, max_length=np.inf):
        self.uuid = uuid.uuid4()
        self.max_length = max_length
        self.transactions = collections.OrderedDict()
        self.veh_app_type = veh_app_type
        self.arrival_size_buffer = TransactionBuffer(max_size=100)
        self.exploded = 0
        self.length = 0
        self.cpu_needs = 0
        self.bw_needs = 0
        self.w = [5,3]
        logger.info('Transaction queue of app. type {} with max length {} is initialized'.format(veh_app_type, max_length))

    def __del__(self):

        if len(self.transactions.keys()):
            ids = list(self.transactions.keys())
            for id in ids:
                del self.transactions[id]

        del self.arrival_size_buffer
        del self


    def remove_multiple_transactions_cloud(self, transaction_list):
        for transaction_id in transaction_list:
            logger.debug('Transaction %s removed', transaction_id)
            data_size = self.transactions[transaction_id].get_data_size()
            self.length -= data_size

            if self.veh_app_type:
                self.cpu_needs -= data_size * Vehicle_applications.get_info(self.veh_app_type)
            else:
                self.cpu_needs -= data_size * Vehicle_applications.get_info(
                    self.transactions[transaction_id].veh_app_type)
            del self.transactions[transaction_id]  # 删除交易

    def remove_multiple_transactions_offload(self, transaction_list):
        for transaction_id in transaction_list:
            logger.debug('Transaction %s removed', transaction_id)
            data_size = self.transactions[transaction_id].get_data_size()
            self.length -= data_size

            self.bw_needs -= data_size
            del self.transactions[transaction_id]

    def remove_multiple_transactions_edge(self,resource, transaction_list):
        total_data = 0
        for transaction_id in transaction_list:
            logger.debug('Transaction %s removed', transaction_id)
            data_size = self.transactions[transaction_id].get_data_size()
            self.length -= data_size
            total_data += data_size /(ct.MB)

            del self.transactions[transaction_id]
        cpu_need, bw_need = self.compute_resource(resource, total_data, types=1)
        self.cpu_needs -= cpu_need
        self.bw_needs -= bw_need


    def arrived(self, transaction, arrival_timestamp):
        transaction_id = transaction.get_uuid()
        transaction_length = transaction.data_size

        self.arrival_size_buffer.add((arrival_timestamp, transaction_length))
        if self.get_length() + transaction_length <= self.max_length:
            self.transactions[transaction_id] = transaction
            self.length += transaction_length
            if self.veh_app_type:
                self.cpu_needs += (((transaction_length / (ct.MB))*(5/8)) / (
                            (1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))) ** 2 / self.w[0]
                self.bw_needs += (((transaction_length / (ct.MB))*(3/8)) / (
                            (1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))) ** 2 / self.w[0]

            self.exploded = max(0, self.exploded-1)
            return True
        else:
            del transaction
            self.exploded = min(10, self.exploded + 1)
            return False

    def arrived_cloud(self,transaction, arrival_timestamp):
        transaction_id = transaction.get_uuid()
        transaction_length = transaction.data_size

        self.arrival_size_buffer.add((arrival_timestamp, transaction_length))
        if self.get_length() + transaction_length <= self.max_length:
            self.transactions[transaction_id] = transaction
            self.length += transaction_length

            if self.veh_app_type:
                self.cpu_needs += transaction_length * Vehicle_applications.get_info(self.veh_app_type)
            else:
                self.cpu_needs += transaction_length * Vehicle_applications.get_info(transaction.veh_app_type)
            self.exploded = max(0, self.exploded - 1)
            return True
        else:
            del transaction
            self.exploded = min(10, self.exploded + 1)
            return False

    def served_offload(self, resource, types=1, silence=True):
        if not silence:
            print("########### compute or offload : inside of transaction_queue.served ##########")
        if resource == 0:
            logger.info('No data to be served')
            return  # 直接返回
        else:
            transaction_to_remove = []
            offloaded_transactions = {}
            served = 0
            to_be_served = resource
            if not silence: print("data size to be offloaded : {}".format(to_be_served))

            for transaction_id, transaction_ob in self.transactions.items():
                transaction_size = transaction_ob.data_size
                if not silence: print("transaction_size : {}".format(transaction_size))

                if to_be_served >= transaction_size:
                    if not silence: print("data size can be served >= task_size case")
                    if not types:
                        offloaded_transactions[transaction_id] = transaction_ob

                    transaction_to_remove.append(transaction_id)
                    to_be_served -= transaction_size
                    served += transaction_size

                elif to_be_served > 0:
                    if not silence: print("data size to be offloaded < task_size case")

                    if self.veh_app_type:
                        transaction_size -= to_be_served

                        if not types:
                            new_transaction = transaction_ob.make_child_transaction(to_be_served)
                            offloaded_transactions[new_transaction.get_uuid()] = new_transaction  # 将新的子交易加入到已转移交易字典中
                        self.length -= to_be_served

                        cpu_txs = to_be_served / ct.MB
                        cpus = (cpu_txs*(5/8) / ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))) ** 2 / self.w[0]
                        bws = (cpu_txs*(3/8) / ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))) ** 2 / self.w[0]
                        self.cpu_needs -= cpus
                        self.bw_needs -= bws
                    served += to_be_served
                    to_be_served = 0
                else:
                    if not silence and not types: print(
                        'All transactions are done in transaction_queue.served(type=0) - offloaded')
                    if not silence and types: print(
                        'All transactions are done in transaction_queue.served(type=1) - computed')
                    break
            resource = served
            self.remove_multiple_transactions_offload(transaction_to_remove)
            if not silence: print("########### task_queue.served ends ###########")
            return resource, offloaded_transactions

    def served(self, resource, types=1, silence=True):

        to_be_served = self.compute_resource(resource, types=0)
        flag = False
        if not silence:
            print("########### compute or offload : inside of transaction_queue.served ##########")
        if to_be_served == 0:
            logger.info('No data to be served')
            return (0.0,0.0),[0,0],None
        else:
            transaction_to_remove = []
            offloaded_transactions = {}
            served = 0
            if not silence: print("data size to be offloaded : {}".format(to_be_served))

            for transaction_id, transaction_ob in self.transactions.items():
                transaction_size = transaction_ob.data_size/(ct.MB)

                if not silence: print("transaction_size : {}".format(transaction_size))

                if to_be_served >= transaction_size:
                    if not silence: print("data size can be served >= task_size case")
                    if not types:
                        offloaded_transactions[transaction_id] = transaction_ob

                    transaction_to_remove.append(transaction_id)

                    to_be_served -= transaction_size

                    served += transaction_size

                elif to_be_served > 0:
                    flag = True
                    if not silence: print("data size to be offloaded < task_size case")
                    transaction_size -= to_be_served

                    if not types:
                        new_transaction = transaction_ob.make_child_transaction(to_be_served)
                        offloaded_transactions[new_transaction.get_uuid()] = new_transaction
                    else:
                        self.transactions[transaction_id].data_size = transaction_size * (ct.MB)
                    self.length -= to_be_served * ct.MB
                    cpu_need, bw_need = self.compute_resource(resource, to_be_served, types=1, flag=1)
                    self.cpu_needs -= cpu_need
                    self.bw_needs -= bw_need
                    served += to_be_served
                    to_be_served = 0
                else:
                    if not silence and not types: print(
                        'All transactions are done in transaction_queue.served(type=0) - offloaded')
                    if not silence and types: print(
                        'All transactions are done in transaction_queue.served(type=1) - computed')
                    break
            self.remove_multiple_transactions_edge(resource, transaction_to_remove)
            if not silence: print("########### task_queue.served ends ###########")
            cpu_need, bw_need = self.compute_resource(resource, served, types=1)
            resource = [
                resource[0] - cpu_need,
                resource[1] - bw_need
            ]
            for i in range(len(resource)):
                if resource[i] < 0:
                    resource[i] = 0

            return (cpu_need, bw_need), resource, offloaded_transactions
    def compute_resource(self, resource,served=0, types=0,flag=0):
        if types == 0:
            sum_resource = 0
            l = 0
            for r in resource:
                sum_resource += (self.w[l] * r) ** (0.5) * ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))
                l += 1
            return sum_resource
        else:
            served_w = [0,0]
            cpu_txs = (self.w[0] * resource[0]) ** (0.5) * ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))
            bw_txs = (self.w[1] * resource[1]) ** (0.5) * ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))
            served_w[0] = served * (5/8)
            served_w[1] = served * (3/8)

            if served < cpu_txs+bw_txs:
                if served_w[0] > cpu_txs:
                    served_w[0] = cpu_txs
                    served_w[1] += served_w[0] - cpu_txs
                if served_w[1] > bw_txs:
                    served_w[1] = bw_txs
                    served_w[0] += served_w[1] - bw_txs
            else:
                served_w[0] = cpu_txs
                served_w[1] = bw_txs


            cpu_needs = (served_w[0] / ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))) ** 2 / self.w[0]
            bw_need = (served_w[1] / ((1 * ct.MB * (1 / (2.35 + 1))) / (ct.MB * 8))) ** 2 / self.w[1]
            return cpu_needs, bw_need





    def served_cloud(self, resource, types=1, silence=True ):

        if not silence:
            print("########### compute or offload : inside of transaction_queue.served ##########")
        if resource == 0:
            logger.info('No data to be served')
            return
        else:
            transaction_to_remove = []
            offloaded_transactions = {}
            served = 0

            if (self.veh_app_type and types):
                to_be_served = resource / Vehicle_applications.get_info(self.veh_app_type, "workload")
            else:
                to_be_served = resource
            if not silence: print("data size to be offloaded : {}".format(to_be_served))

            for transaction_id, transaction_ob in self.transactions.items():
                transaction_size = transaction_ob.data_size

                workload = Vehicle_applications.get_info(transaction_ob.get_app_type(), "workload")
                if (not self.veh_app_type and types):
                    transaction_size *= workload
                if not silence: print("transaction_size : {}".format(transaction_size))

                if to_be_served >= transaction_size:
                    if not silence: print("data size can be served >= task_size case")
                    if not types:
                        offloaded_transactions[transaction_id] = transaction_ob

                    transaction_to_remove.append(transaction_id)

                    to_be_served -= transaction_size

                    served += transaction_size

                elif to_be_served > 0:
                    if not silence: print("data size to be offloaded < task_size case")

                    if self.veh_app_type:
                        transaction_size -= to_be_served

                        if not types:
                            new_transaction = transaction_ob.make_child_transaction(to_be_served)
                            offloaded_transactions[new_transaction.get_uuid()] = new_transaction  # 将新的子交易加入到已转移交易字典中
                        else:
                            self.transactions[transaction_id].data_size = transaction_size  # 更新当前任务的剩余数据量
                        self.length -= to_be_served
                        self.cpu_needs -= to_be_served * Vehicle_applications.get_info(self.veh_app_type)


                    else:
                        transaction_size /= workload
                        to_be_served = int(to_be_served / workload)
                        transaction_size -= to_be_served
                        self.transactions[transaction_id].data_size = transaction_size
                        self.length -= to_be_served
                        self.cpu_needs -= to_be_served * workload
                        to_be_served *= workload
                    served += to_be_served
                    to_be_served = 0
                else:
                    if not silence and not types: print('All transactions are done in transaction_queue.served(type=0) - offloaded')
                    if not silence and types: print('All transactions are done in transaction_queue.served(type=1) - computed')
                    break

            if types and self.veh_app_type:
                resource = served * Vehicle_applications.get_info(self.veh_app_type, "workload")
            else:
                resource = served
            self.remove_multiple_transactions_cloud(transaction_to_remove)
            if not silence: print("########### task_queue.served ends ###########")
            return resource, offloaded_transactions


    def mean_arrival(self, t, interval=10, scale=1):
        result = 0
        for time, data_size in self.arrival_size_buffer.get_buffer():
            if time > t - interval:
                result += data_size
            else:
                break
        return result / min(t + 1, interval) /scale

    def last_arrival(self, t, scale):
        last_data = self.arrival_size_buffer.get_last_obs()
        if last_data:
            time, data_size = last_data
            if time == t:
                return data_size/scale
        return 0

    def get_uuid(self):
        return self.uuid.hex


    def get_length(self, scale=1):
        return self.length / scale


    def get_cpu_needs(self, scale=1):
        return self.cpu_needs / scale


    def get_status(self):
        return self.transactions, self.exploded

    def get_transactions(self):
        return self.transactions


    def get_max(self, scale=1):
        return self.max_length / scale


    def is_exploded(self):
        return self.exploded
