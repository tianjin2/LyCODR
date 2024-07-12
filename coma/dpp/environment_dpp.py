import collections

import collections
from coma.envs.servernode_w_appqueue_w_appinfo_cores import ServerNode as Edge
from coma.envs.cloudnode_w_totalqueue_cores import CloudNode as Cloud
from coma.envs.Vehicle_applications import *
from coma.envs.environment import Environment
from coma.envs.constant import *
from coma.envs.channels import *
from coma.envs.rl_networks.utils import *
from coma.envs.virtual_queue import VirtualQueue
import gym
from gym import error, spaces
from gym.utils import seeding


class MEC_v1(Environment):
    def __init__(self, transaction_rate, *veh_applications, use_beta=True, empty_reward=True, cost_type=1,
                 max_episode_steps=5000, time_stamp=0, shard_list=None,agent_num = 10, edge_capability=200, edge_bws=200):
        super().__init__()
        self.shard_list = shard_list if shard_list is not None else {}
        self.agent_num = agent_num
        self.serverNodes = dict()
        self.cloudNodes = dict()
        self.accounts = dict()
        self.links = list()
        self.timestamp = time_stamp
        self.silence = True
        self.veh_applications = veh_applications

        self.transaction_rate = transaction_rate
        self.reset_info = list()  # Reset information
        self.use_beta = use_beta  # Whether to perform offloading
        self.cost_type = cost_type  # Penalty item weights in DPP
        self.max_episode_steps = max_episode_steps
        self.before_arrival = None
        self.sum_cpu = edge_capability
        self.sum_bw = edge_bws
        self.lambdas = []
        self.resource_list = VirtualQueue(veh_applications, self.sum_cpu + self.sum_bw)

        channel = WIRED
        self.t = 0
        # Computing power of computational edge servers and cloud services
        cloud_capability = NUM_CLOUD_CORES * NUM_CLOUD_CLOCK_FREQUENCY * GHZ

        self.reset_info.append((edge_capability, edge_bws, cloud_capability, channel))  # Store reset information
        state = self.init_linked_pair(edge_capability, edge_bws, cloud_capability, channel)
        self.serverNodes = dict()  # Reinitialise the server dictionary
        self.cloudNodes = dict()  # Reinitialise the cloud dictionary
        self.links = list()  # Reinitialise the linked list
        self.accounts = dict()
        self._seed()  # Setting random seeds

    def init_linked_pair(self, edge_capability, edge_bws, cloud_capability, channel):
        for shard_id in self.shard_list.values():
            EdgeNode = self.add_server(edge_capability, edge_bws, shard_id)  # adding edge nodes
            EdgeNode.make_veh_application_queues(*self.veh_applications)
            # adding cloud nodes
            CloudNode = self.add_cloud(cloud_capability)
            # Add edge nodes and cloud nodes to communication channels
            self.add_link(EdgeNode, CloudNode, channel)

        # Getting the state of the environment
        states = self.get_status(scale=ct.MB)
        self.state_dim = len(states[0])
        self.action_dim += len(self.shard_list) * 2
        if self.use_beta:
            self.action_dim = len(self.shard_list) * 3
        return states

    def add_server(self, cap, bws, uuid_node):
        EdgeNode = Edge(uuid_node=uuid_node, computational_capability=cap, bws=bws, shard_list=self.shard_list,
                        is_random_generating=True)  # 创建一个边缘节点
        EdgeNode.set_account(50)
        self.accounts[uuid_node] = EdgeNode.get_account()
        self.serverNodes[EdgeNode.get_uuid()] = EdgeNode
        return EdgeNode

    def add_cloud(self, cap):
        CloudNode = Cloud(cap)
        self.cloudNodes[CloudNode.get_uuid()] = CloudNode
        return CloudNode

    # Add links between edge nodes and cloud nodes
    def add_link(self, serverNode, cloudNode, up_channel, down_channel=None):
        up_channel = Channel(up_channel)  # Create an uplink communication channel object

        if not down_channel:
            down_channel = Channel(up_channel)
        else:
            down_channel = Channel(down_channel)
        serverNode.links_to_higher[cloudNode.get_uuid()] = {
            'node': cloudNode,
            'channel': up_channel
        }
        cloudNode.links_to_lower[serverNode.get_uuid()] = {
            'node': serverNode,
            'channel': down_channel
        }
        # Add link information
        self.links.append((serverNode.get_uuid(), cloudNode.get_uuid()))

    # setting random seed
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_action_dim(self):
        return self.action_dim

    def get_observation_dim(self):
        return self.state_dim

    def get_number_of_apps(self):
        return len(self.veh_applications)

    def __del__(self):
        for k in list(self.serverNodes.keys()):
            del self.serverNodes[k]
        for k in list(self.cloudNodes.keys()):
            del self.cloudNodes[k]
        del self.links
        del self.veh_applications


    def reset(self, empty_reward=True, rand_start=0):
        # Save the current environment configuration
        transaction_rate = self.transaction_rate
        veh_applications = self.veh_applications
        use_beta = self.use_beta
        cost_type = self.cost_type
        time_stamp = self.timestamp
        shard_list = self.shard_list
        self.__del__()
        # Reinitialise the environment
        self.__init__(transaction_rate, *veh_applications, use_beta=use_beta, empty_reward=empty_reward,
                      cost_type=cost_type, time_stamp=time_stamp, shard_list=shard_list)

        for reset_info in self.reset_info:
            self.init_linked_pair(*reset_info)

        self.before_arrival = self._get_edge_qlength(scale=ct.MB)

        # Get current observation status
        reset_state = self.get_status(scale=ct.MB)

        return reset_state, self.resource_list.get_length()

    def get_status(self, scale=1):
        edge_state, temp_state = list(), list()
        state = list()
        # Iterate over the edge nodes to get the state and add it to the edge_state list
        for serverNode in self.serverNodes.values():
            # Get the status of a single edge node
            edge_state = serverNode._get_obs(self.timestamp, scale=scale)
            temp_state += edge_state
            if self.use_beta:
                for cloudNode in serverNode.get_higher_node_ids():
                    temp_state += self.cloudNodes[cloudNode]._get_obs(self.timestamp, scale=scale)
            state.append(temp_state)
            temp_state = list()

        return np.array(state)

    def step(self, actions, t=0, use_beta=True, generate=True):
        # if self.t >= 5000 and self.t <6000:
        #     self.transaction_rate = 8
        #     print(self.transaction_rate)
        # elif self.t >= 6000:
        #     self.transaction_rate = 2
        self.t += 1
        q0, failed_to_generate, q1 = self._step_generation()
        action_alpha, action_beta, action_beta_offload, usage_ratio = list(), list(), list(), list()

        # Split the action vector into alpha and beta parts
        if self.use_beta:
            action_beta_offload = actions[self.agent_num*2:self.agent_num*3]
        action_alpha = actions[:self.agent_num]
        action_beta = actions[self.agent_num:self.agent_num * 2]

        self.alloc_resource(self.sum_cpu, self.sum_bw, action_alpha, action_beta,action_beta_offload, t)
        # Perform alpha and beta steps to process edge and cloud transactions
        used_edge_cpus, used_edge_bws, init_state, q2 = self._step_local(action_alpha, action_beta)
        used_cloud_bws, used_cloud_cpus, new_state, q_last = self._step_cloud(action_beta_offload)
        cost = self.get_cost(used_edge_cpus, used_cloud_cpus, used_edge_bws, used_cloud_bws, q0, q2)
        self.timestamp += 1
        if self.timestamp == self.max_episode_steps:
            return new_state, cost, [1 for i in range(len(self.serverNodes.keys()))],self.resource_list.get_length()
        return new_state, cost, [0 for i in range(len(self.serverNodes.keys()))], self.resource_list.get_length()

    def alloc_resource(self, computational_capability, bw, action_alpha, action_beta,action_beta_offload, t):
        alpha = action_alpha
        beta = action_beta
        offload = action_beta_offload
        resource_sum = [
            [alpha[i] * computational_capability for i in range(len(alpha))],
            [beta[i]+offload[i] * bw / ct.KBPS for i in range(len(action_beta))],
        ]
        self.resource_list.allocation_resource(resource_sum, t)

    def _step_local(self, action_alpha, action_beta):
        used_egde_cpus = collections.defaultdict(float)
        used_edge_bws = collections.defaultdict(float)
        action_alpha = action_alpha
        action_beta = action_beta
        # Calculate resource allocation per shard
        shard_list = dict()
        for shard_name in self.shard_list:
            shard_list[shard_name] = self.shard_list[shard_name].hex
        res_allocs = dict(zip(shard_list.values(), zip(action_alpha, action_beta)))

        for index, shard_id in enumerate(shard_list.values()):
            if res_allocs[shard_id] == (0., 0.) or (shard_id not in self.serverNodes.keys()):
                pass
            else:
                resource = [
                    [x[0] for x in res_allocs.values()][index] * self.sum_cpu,
                    [x[1] for x in res_allocs.values()][index] * self.sum_bw,
                ]
                used_egde_cpus[shard_id], used_edge_bws[shard_id] = self.serverNodes[shard_id].do_transactions(resource)
        state = self.get_status(scale= ct.MB)
        after_length = self._get_edge_qlength(scale= ct.MB)
        if self.timestamp % 1000 == 0:
            print("alpha", 1 - sum(action_alpha), "beta", 1 - sum(action_beta))  # 打印剩余资源比例
        return used_egde_cpus, used_edge_bws, state, after_length

    def _step_cloud(self, action_beta):
        # Initialise the dictionary used to record transactions transmitted by each edge node
        used_txs = collections.defaultdict(list)
        used_edge_bws = collections.defaultdict(float)
        # Initialise the dictionary used to record offload transactions
        transactions_to_be_offloaded = collections.defaultdict(dict)
        # Initialise the dictionary used to record CPU resources used by cloud nodes
        used_cloud_cpus = collections.defaultdict(float)
        # Talk about transforming action arrays into one-dimensional shapes.
        action = action_beta
        # Get the length of the edge queue before the unload action
        q_before = self._get_edge_qlength(scale=ct.MB)
        # Iterate over each edge node and its actions, offloading transactions to upper nodes, (needs to be changed, here it just corresponds to an action)
        for serverNode, beta in list(zip(self.serverNodes.values(), action)):
            higher_nodes = serverNode.get_higher_node_ids()
            # Perform operations on each upper node
            for higher_node in higher_nodes:
                # offload transactions and record information
                used_tx, transaction_to_be_offloaded, failed = serverNode.offload_transactions(beta, higher_node)
                used_edge_bws[serverNode.get_uuid()] += used_tx / serverNode.sample_channel_rate(higher_node)
                # Transmission transaction records updated
                used_txs[higher_node].append(used_tx)
                # Update the record of transactions to be offloaded
                transactions_to_be_offloaded[higher_node].update(transaction_to_be_offloaded)
        # Get the length of the edge queue after the offload action
        q_last = self._get_edge_qlength(scale=ct.MB)
        # Get the status information of the offload action
        s1 = self.get_status(scale=ct.MB)

        i = 0
        for cloudNode_id, cloudNode in self.cloudNodes.items():
            cloudNode.offloaded_transactions(transactions_to_be_offloaded[cloudNode_id], self.timestamp)
            used_cloud_cpus[cloudNode_id] = cloudNode.do_transactions()
            s2 = self.get_status(scale=ct.MB)

            used_cloud_cpus[cloudNode_id] = (s2 - s1)[i][-2]
            i += 1

        state = self.get_status(scale=ct.MB)

        if self.timestamp % 1000 == 0:
            print("beta", 1 - sum(action))
        # Returns the cloud CPU resources used, the new state and the last queue length
        return used_edge_bws, used_cloud_cpus, state, q_last


    def get_cost(self, used_edge_cpus, used_cloud_cpus, used_edge_bws, used_cloud_bws, before, after, failed_to_offload=0, failed_to_generate=0):
        # Calculate virtual queues
        resource_gas = max(self.resource_list.get_length() - (self.sum_cpu + self.sum_bw), 0)

        edge_drift_cost = sum((np.array(after))**2) * 0.5 - sum((np.array(before))**2) * 0.5

        resource_drift_cost = 2 * self.resource_list.get_length() * (resource_gas) + (resource_gas) ** 2

        self.resource_list.set_length(self.resource_list.get_length() + resource_gas)
        # resource_drift_cost = sum(2 * ())
        cpu_computation_cost = 0
        bw_computation_cost = 0
        cloud_payment_cost = 0
        # Calculate the cost corresponding to the CPU used by each edge node
        for used_edge_cpu in used_edge_cpus.values():
            cpu_computation_cost += used_edge_cpu/200
        for used_edge_bw in used_edge_bws.values():
            bw_computation_cost += used_edge_bw/200
        for used_cloud_cpu in used_cloud_cpus.values():
            bw_computation_cost += used_cloud_cpu/200
        for used_cloud_bw in used_cloud_bws.values():
            cloud_payment_cost += used_cloud_bw/200

        return (edge_drift_cost + resource_drift_cost + self.cost_type * (cpu_computation_cost + bw_computation_cost + cloud_payment_cost))

    def _get_edge_qlength(self, scale=1):
        qlengths = list()
        for node in self.serverNodes.values():
            qlength = 0
            for _, queue in node.get_queue_list():
                qlength += queue.get_length(scale)  # Add application queue length per node
            qlengths.append(qlength)
        return qlengths


    def _step_generation(self):
        initial_qlength = self._get_edge_qlength(scale=ct.MB)
        failed_to_generates = 0
        if not self.silence: print("###### random transaction generation start! ######")
        # Generate random transactions for each edge node and handle generation failures
        for serverNode in self.serverNodes.values():
            arrival_size, failed_to_generate = serverNode.random_transaction_generation(self.transaction_rate,
                                                                                        self.timestamp,
                                                                                        *self.veh_applications)
            failed_to_generates += failed_to_generate
        if not self.silence: print("###### random transaction generation ends! ######")

        # Get the length of the transaction queue after generating a transaction
        after_qlength = self._get_edge_qlength(scale=ct.MB)
        self.lambdas = [after - befor for after, befor in zip(after_qlength, initial_qlength)]

        return initial_qlength, failed_to_generates, after_qlength