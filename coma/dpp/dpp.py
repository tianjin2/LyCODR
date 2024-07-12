from scipy.optimize import minimize
from coma.envs.constant import *
import numpy as np
import math


class KKT_actor(object):
    def __init__(self, f_C, f_B, B, Z_length,cost_type, agent_num, scale=1):
        self.f_C = f_C
        self.f_B = f_B
        self.B = B
        self.cost_type = cost_type
        self.scale = scale
        self.S_B = 1 * MB
        self.S_T = 500 * BYTE
        self.T_cons = 8
        self.R_r = 2.35
        self.w_d = [5, 3]
        self.Z_length = Z_length
        self.agent_num = agent_num
        self.txs = 2000
    def optimize(self, state, resource, agent_num):
        states = np.array(state)[:self.agent_num, :-3]
        arrivals = [arr[1] for arr in states]
        q_lengths = [q[2] for q in states]
        cpu_need = [c[3] for c in states]
        bw_need = [b[4] for b in states]
        resource_length = resource

        action = list()
        for i in range(self.agent_num):
            action.append(1/self.agent_num)
        for i in range(self.agent_num*2):
            action.append(1/(agent_num*2))

        def compute_zd(resource_length):
            z_k = max(resource_length - (self.f_C + self.f_B), 0)
            return z_k

        def objective(action):
            edge_drift_cost = 0
            B_E_All = 0

            for i in range(agent_num):
                B_E = 0
                B_C = 0
                B_E += ((action[i] * self.w_d[0] * self.f_C) ** 0.5) * self.S_B * (
                                      1 / (self.R_r + 1)) / (self.S_T * self.T_cons) /self.txs
                B_E += ((action[i + agent_num] * self.w_d[1] * self.f_B) ** 0.5) * self.S_B * (
                                      1 / (self.R_r + 1)) / (self.S_T * self.T_cons) /self.txs
                B_E_All += B_E
                B_C = self.B * action[i + agent_num * 2] / MB

                edge_drift_cost += 2*(q_lengths[i]) * ((arrivals[i] - (B_C + B_E) )) + ((arrivals[i] - (B_C + B_E) ))**2
                B_E_All += B_C

            edge_drift_cost += self.Z_length * compute_zd(resource_length)
            edge_computation_cost = (sum([action[i] * self.f_C for i in range(agent_num)]) + sum([action[i+agent_num] * self.f_B  for i in range(agent_num)])) / 200
            could_cost = sum(action[i+agent_num*2] * self.f_B  for i in range(agent_num)) / 200
            reward = 1 * B_E_All

            return (edge_drift_cost + self.cost_type * (edge_computation_cost + could_cost - reward))

        # Constraints
        consts = []
        def constrain_v1(x):
            return (1 - sum(x[:self.agent_num]))

        consts += [{'type':'ineq', 'fun': constrain_v1}]

        def constrain_v2(x):
            return (1 - sum(x[self.agent_num:self.agent_num*3]))

        consts += [{'type': 'ineq', 'fun': constrain_v2}]


        def const_function(k):
            def constraint(x):

                return x[k]
            return constraint

        for i in range(agent_num*3):
            consts += [{'type':'ineq', 'fun':const_function(i)}]

        def const_function2(k):
            def constraint(x):
                B_E = (x[k] * self.w_d[0] * self.f_C + x[k+self.agent_num] * self.w_d[1] * self.f_B ) * self.S_B * (1 / (self.R_r + 1)) / (
                            self.S_T * self.T_cons) /self.txs
                B_C = self.B * x[k+self.agent_num*2] / MB
                return (q_lengths[k] + arrivals[k] - (B_C + B_E))
            return constraint
        for i in range(self.agent_num):
            consts += [{'type':'ineq', 'fun':const_function2(i)}]

        cons = tuple(consts)
        action = np.array(action)
        solution = minimize(objective, action, bounds=[(0, 1)] * len(action), constraints=cons, options={'maxiter':999}, method="SLSQP")

        if not solution.success:
            import pdb; pdb.set_trace()
            print("FALSE")

        E = solution.x
        action = np.array(list(E[:self.agent_num]) + list(E[self.agent_num:self.agent_num*2]) + list(E[self.agent_num*2:self.agent_num*3]))
        return (action >= 0) * action