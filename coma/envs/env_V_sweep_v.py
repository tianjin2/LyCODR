# -*- coding: utf-8 -*-
import collections
from coma.envs.servernode_w_appqueue_w_appinfo_cores import ServerNode as Edge
from coma.envs.cloudnode_w_totalqueue_cores import CloudNode as Cloud
from coma.envs.Vehicle_applications import *
from coma.envs.channels import *
from coma.envs.rl_networks.utils import *
from coma.envs.virtual_queue import VirtualQueue
import gym
from gym import error, spaces
from gym.utils import seeding


class MEC(gym.Env):
    def __init__(self, transaction_rate=8, veh_applications=(SPEECH_RECOGNITION, NLP, FACE_RECOGNITION), use_beta=True,
                 empty_reward=True, cost_type=COST_TYPE,
                 max_episode_steps=5000, time_stamp=0, shard_list=None, edge_capability=200, edge_bws=200):
        super().__init__()

        self.shard_list = shard_list if shard_list is not None else {}
        # Initialise various states and properties
        self.state_dim = 0
        self.action_dim = 0
        self.serverNodes = dict()
        self.cloudNodes = dict()
        self.accounts = dict()
        self.links = list()
        self.timestamp = time_stamp
        self.silence = True
        self.veh_applications = veh_applications
        self.transaction_rate = transaction_rate
        self.reset_info = list()  # Reset information
        self.use_beta = use_beta   # Whether to perform offloading
        self.cost_type = cost_type  # Penalty item weights in DPP
        self.max_episode_steps = max_episode_steps
        self.before_arrival = None
        self.sum_cpu = edge_capability
        self.sum_bw = edge_bws

        self.lambdas = []
        channel = WIRED

        # Computing power of computational edge servers and cloud services
        cloud_capability = NUM_CLOUD_CORES * NUM_CLOUD_CLOCK_FREQUENCY * GHZ
        self.make_veh_resource_queue(self.veh_applications, self.sum_cpu, self.sum_bw)
        self.reset_info.append((edge_capability, edge_bws, cloud_capability, channel))  # 存储重置信息
        state = self.init_linked_pair(edge_capability, edge_bws, cloud_capability, channel)
        self.t = 0
        # Calculate the observation dimension
        self.obs_dim = len(state[0]) - 3
        # Define upper and lower limits of the behavioural space
        high = np.ones(self.action_dim)
        low = - high
        self.action_space = spaces.Box(low, high)  # First behavioural space Continuous space
        self.action_dim = 0
        # Define upper and lower limits of the observation space
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.resource_list = VirtualQueue(veh_applications, self.sum_cpu + self.sum_bw)
        self.serverNodes = dict()  # Reinitialise the server dictionary
        self.cloudNodes = dict()  ## Reinitialise the cloud dictionary
        self.links = list()  # Reinitialise the linked list
        self.accounts = dict()
        self._seed()  # Setting random seeds

    def __del__(self):
        for k in list(self.serverNodes.keys()):
            del self.serverNodes[k]
        for k in list(self.cloudNodes.keys()):
            del self.cloudNodes[k]
        del self.links
        del self.veh_applications

    def make_veh_resource_queue(self, veh_app_types, computational_capability, bw):
        self.resource_list = VirtualQueue(veh_app_types, computational_capability + bw)
        return

    def init_linked_pair(self, edge_capability, edge_bws, cloud_capability, channel):
        for shard_id in self.shard_list.values():
            EdgeNode = self.add_server(edge_capability, edge_bws, shard_id)  # 添加边缘节点
            EdgeNode.make_veh_application_queues(*self.veh_applications)
            # adding cloud nodes
            CloudNode = self.add_cloud(cloud_capability)
            # Add edge nodes and cloud nodes to communication channels
            self.add_link(EdgeNode, CloudNode, channel)

        # Getting the state of the environment
        states = self._get_obs(scale=ct.MB)

        self.state_dim = len(states[0])
        self.action_dim += len(self.shard_list) * 2
        if self.use_beta:
            self.action_dim = len(self.shard_list) * 3
        return states

    def add_server(self, cap, bws, uuid_node):
        EdgeNode = Edge(uuid_node=uuid_node, computational_capability=cap, bws=bws, shard_list=self.shard_list,
                        is_random_generating=True)
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
        return self.obs_dim


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
        self.__init__(transaction_rate, veh_applications, use_beta=use_beta, empty_reward=empty_reward,
                      cost_type=cost_type, time_stamp=time_stamp, shard_list=shard_list)
        for reset_info in self.reset_info:
            self.init_linked_pair(*reset_info)
        self.before_arrival = self._get_edge_qlength(scale=ct.MB)

        _, failed_to_generate, _ = self._step_generation()
        # Get current observation status
        reset_state = self._get_obs(scale=ct.MB)
        for i in range(len(reset_state)):
            reset_state[i][1] = self.before_arrival[i]  # Update queue length information
        reset_state = reset_state[:, :-3]
        reset_state[:, -1] = 0.0  # Set the last state value to 0. This state value is represented as the average arrival rate of transactions at the cloud node
        return reset_state


    def _get_obs(self, scale=ct.MB):
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
        # if self.t >5000 and self.t <= 6000:
        #     self.transaction_rate = 10
        # elif self.t > 6000:
        #     self.transaction_rate = 2

        q0 = np.array(self.before_arrival)
        start_state = self._get_obs(scale=ct.MB)
        # Get the length of each shard of the queue in which the action is performed
        q1 = np.array(self._get_edge_qlength(scale=ct.MB))
        # Split the action vector into alpha and beta parts
        if self.use_beta:
            action_alpha = actions[0].reshape(1,-1)
            action_beta = actions[1].reshape(1,-1)
            action_beta_offload = actions[2].reshape(1,-1)
            action_bw = np.hstack((action_beta, action_beta_offload))
            # Apply the softmax function to the beta part to make it a probability distribution
            action_beta_bw = softmax_1d(action_bw)
            action_beta = action_beta_bw.flatten()[:len(self.serverNodes)].reshape(1,-1)
            action_beta_offload = action_beta_bw.flatten()[len(self.serverNodes):].reshape(1,-1)

        else:
            action_alpha = actions[0].reshape(1,-1)
            action_beta = actions[1].reshape(1,-1)
            action_beta = softmax_1d(action_beta)

        action_alpha = softmax_1d(action_alpha)
        self.alloc_resource(self.sum_cpu, self.sum_bw, action_alpha, action_beta, action_beta_offload, t)
        # Perform alpha and beta steps to process edge and cloud transactions
        used_edge_cpus, used_edge_bws, inter_state = self._step_local(action_alpha, action_beta)
        used_cloud_bws, used_cloud_cpus, new_state, q_last = self._step_cloud(action_beta_offload)
        self.before_arrival = q_last
        self.t += 1
        _, failed_to_generate, _ = self._step_generation()

        new_state = self._get_obs(scale=ct.MB)

        cost,total_consumption = self.get_cost(used_edge_cpus, used_cloud_cpus, used_edge_bws, used_cloud_bws, q0, q_last - q1,
                             cost_type=self.cost_type)
        new_state = [row[:-3].tolist() for row in new_state]

        for index,cpu_used in enumerate(used_cloud_cpus.values()):
            new_state[index][-1] = cpu_used

        self.timestamp += 1

        if self.timestamp % 1000 ==0:
            print(self.timestamp)
        if self.timestamp == self.max_episode_steps:
            return new_state, -cost, [1 for i in range(len(self.serverNodes.keys()))], {"cloud_cpu_used": start_state[-1]}
        return new_state, -cost, [0 for i in range(len(self.serverNodes.keys()))], {"cloud_cpu_used": start_state[-1]}

    def alloc_resource(self, computational_capability, bw, action_alpha, action_beta, action_offload, t):
        alpha = action_alpha.flatten().reshape(1, -1)
        beta = action_beta.flatten().reshape(1, -1)
        offload = action_offload.flatten().reshape(1, -1)
        resource_sum = [
            (sum(alpha) * computational_capability).tolist(),
            (sum(beta + offload) * bw).tolist(),
        ]

        self.resource_list.allocation_resource(resource_sum, t)

    def _step_local(self, action_alpha, action_beta):
        used_egde_cpus = collections.defaultdict(float)
        used_edge_bws = collections.defaultdict(float)
        action_alpha = action_alpha.flatten().reshape(1, -1)
        action_beta = action_beta.flatten().reshape(1, -1)

        # Calculate resource allocation per shard
        shard_list = dict()
        for shard_name in self.shard_list:
            shard_list[shard_name] = self.shard_list[shard_name].hex

        res_allocs = dict(zip(shard_list.values(), zip(action_alpha.flatten(), action_beta.flatten())))
        for index, shard_id in enumerate(shard_list.values()):
            if res_allocs[shard_id] == (0., 0.) or (shard_id not in self.serverNodes.keys()):
                pass
            else:
                resource = [
                    [x[0] for x in res_allocs.values()][index] * self.sum_cpu,
                    [x[1] for x in res_allocs.values()][index] * self.sum_bw,
                ]
                used_egde_cpus[shard_id], used_edge_bws[shard_id] = self.serverNodes[shard_id].do_transactions(resource)
        state = self._get_obs(scale=ct.MB)
        if self.timestamp % 1000 == 0:
            print("alpha", 1 - sum(sum(action_alpha)), "beta", 1 - sum(sum(action_beta)))  # 打印剩余资源比例
        return used_egde_cpus, used_edge_bws, state


    def _step_cloud(self, action_beta):
        # Initialise the dictionary used to record transactions transmitted by each edge node
        used_txs = collections.defaultdict(list)
        used_edge_bws = collections.defaultdict(float)
        # Initialise the dictionary used to record offload transactions
        transactions_to_be_offloaded = collections.defaultdict(dict)
        # Initialise the dictionary used to record CPU resources used by cloud nodes
        used_cloud_cpus = collections.defaultdict(float)
        # Talk about transforming action arrays into one-dimensional shapes.
        action = action_beta.flatten().reshape(1, -1)
        action = action.flatten()
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
        s1 = self._get_obs(scale=ct.MB)
        i = 0
        for cloudNode_id, cloudNode in self.cloudNodes.items():
            cloudNode.offloaded_transactions(transactions_to_be_offloaded[cloudNode_id], self.timestamp)
            s2 = self._get_obs(scale=ct.MB)
            used_cloud_cpus[cloudNode_id] = cloudNode.do_transactions()
            used_cloud_cpus[cloudNode_id] = (s2-s1)[i][-2]
            i+=1


        state = self._get_obs(scale=ct.MB)
        if self.timestamp % 1000 == 0:
            print("beta", 1 - sum(action))
        # Returns the cloud CPU resources used, the new state and the last queue length
        return used_edge_bws, used_cloud_cpus, state, q_last


    def get_cost(self, used_edge_cpus, used_cloud_cpus, used_edge_bws, used_cloud_bws, before, bt, cost_type):
        cost_type = 10
        # Calculate virtual queues
        lamd = np.array(self.lambdas)
        resource_gas = max(self.resource_list.get_length() - (self.sum_cpu + self.sum_bw ), 0)
        edge_drift_costs = 2 * (before ) * (lamd + bt) + ((lamd + bt) ) ** 2
        edge_drift_cost = sum(edge_drift_costs)
        resource_drift_cost = 2 * self.resource_list.get_length() * (resource_gas) + (resource_gas) ** 2

        resource_drift_costs = [resource_drift_cost for i in range(len(self.serverNodes.keys()))]
        self.resource_list.set_length(self.resource_list.get_length() + resource_gas)
        cpu_computation_cost = 0
        bw_computation_cost = 0
        cloud_payment_cost = 0
        # Calculate the cost corresponding to the CPU used by each edge node
        for used_edge_cpu in used_edge_cpus.values():
            cpu_computation_cost += used_edge_cpu /200
        for used_edge_bw in used_edge_bws.values():
            bw_computation_cost += used_edge_bw /200
        for used_cloud_cpu in used_cloud_cpus.values():
            bw_computation_cost += used_cloud_cpu /200
        for used_cloud_bw in used_cloud_bws.values():
            cloud_payment_cost += used_cloud_bw / 200

        if self.timestamp % 1000 == 0:
            print("virtual_queue: ", self.resource_list.get_length())
            print("used cpu edge: ", sum(used_edge_cpus.values()))
            print("used bw edge: ", sum(used_edge_bws.values()) + sum(used_cloud_bws.values()))
            print("used cpu cloud: ", sum(used_cloud_cpus.values()))
            print("edge power: ", cpu_computation_cost + bw_computation_cost)
            print("cloud power: ", cloud_payment_cost)
            print("bt:", sum(np.array(bt)))
            print("power * cost : ", cost_type * (cpu_computation_cost + bw_computation_cost + cloud_payment_cost))
            print("cost : ", edge_drift_cost + resource_drift_cost + cost_type * (
                        cpu_computation_cost + bw_computation_cost + cloud_payment_cost+sum(np.array(bt))))
            print("rew : ", -(edge_drift_cost + resource_drift_cost + cost_type * (
                        cpu_computation_cost + bw_computation_cost + cloud_payment_cost+sum(np.array(bt)))))
        return (edge_drift_costs + resource_drift_costs + cost_type * ((
                    np.array(list(used_edge_cpus.values()))/200 + np.array(list(used_edge_bws.values()))/200 + np.array(list(used_cloud_cpus.values()))/200 +np.array(list(used_cloud_bws.values()))/200
                    )+5*np.array(bt))),sum(used_edge_cpus.values())+sum(used_edge_bws.values()) + sum(used_cloud_bws.values())+sum(used_cloud_cpus.values())


    def _get_edge_qlength(self, scale=1):
        qlengths = list()
        for node in self.serverNodes.values():
            qlength = 0
            for _, queue in node.get_queue_list():
                length = queue.get_length(scale)  # Add application queue length per node
                qlength += length
            qlengths.append(qlength)

        return qlengths

    def _step_generation(self):
        initial_qlength = self._get_edge_qlength(scale=ct.MB)
        failed_to_generates = 0
        if not self.silence: print("###### random transaction generation start! ######")
        #  Generate random transactions for each edge node and handle generation failures
        for serverNode in self.serverNodes.values():
            arrival_size, failed_to_generate = serverNode.random_transaction_generation(self.transaction_rate,
                                                                                        self.timestamp,
                                                                                        *self.veh_applications)
            failed_to_generates += failed_to_generate
        if not self.silence: print("###### random transaction generation ends! ######")

        # Get the length of the transaction queue after generating a transaction
        after_qlength = self._get_edge_qlength(scale=ct.MB)
        self.lambdas = [after - befor for after,befor in zip(after_qlength, initial_qlength)]
        return initial_qlength, failed_to_generates, after_qlength

