import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from coma.algos.base_coma import RLAlgorithm
from coma.misc.serializable import Serializable
from coma.utils import logger
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt





class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3_mean = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = F.tanh(self.fc3_mean(x))
        return mean




class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic, self).__init__()
        input_dim = 1 + state_dim  + action_dim

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # Forward propagation method for models to compute the data flow from inputs to outputs
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



device = torch.device('cpu')
class COMA(RLAlgorithm, Serializable):
    def __init__(
            self,
            base_kwargs,
            env,
            pool,

            agent_num,
            plotter=None,
            lr_c=1e-4,
            lr_a=5e-3,
            tau=0.01,
            gamma=0.99,
            target_update_steps=1,
            scale_reward=200,
            initial_exploration_done=False,
            save_full_state=False,
    ):
        Serializable.quick_init(self, locals())
        super(COMA, self).__init__(**base_kwargs)

        # Storage environment objects
        self._env = env
        # Storage experience replay pool
        self._pool = pool
        # Store plotters for visualisation
        self._plotter = plotter

        # Store reward scaling factors
        self._scale_reward = scale_reward

        # Discount factor
        self._gamma = gamma

        #  the number of agents
        self._agent_num = agent_num

        # Number of steps in the target network update
        self.target_update_steps = target_update_steps

        self._initial_exploration_done = initial_exploration_done

        self._save_full_state = save_full_state

        self.loss = []
        # state dimension
        self._state_dim = 6
        # action dimension
        self._action_dim = 3
        self.action_var = torch.full((self._action_dim,), 0.5 * 0.5).to(device)

        # Store actor information
        self.actors = [Actor(self._state_dim, self._action_dim) for _ in range(agent_num)]
        # Storing critic information
        self.critics = [Critic(agent_num,self._state_dim,self._action_dim) for _ in range(agent_num)]
        # Create a Target Critic object, initially with the same parameters as the main Critic.
        self.critic_targets = [Critic(agent_num,self._state_dim,self._action_dim) for _ in range(agent_num)]
        for critic_target, critic in zip(self.critic_targets, self.critics):
            critic_target.load_state_dict(critic.state_dict())

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=lr_c) for i in range(agent_num)]
        self.count = 0

    def get_agent_num(self):
        return self._agent_num

    def get_action_dim(self):
        return self._action_dim

    def get_observation_dim(self):
        return self._state_dim


    def get_actions_random(self,observations,current_iteration,path_length,deterministic=True):
        return np.random.uniform(-1., 1., (self._agent_num,self._action_dim)), np.zeros((self._agent_num,self._action_dim)),None

    def get_actions_avg(self,observations,current_iteration,path_length,deterministic=True):
        ac = np.zeros((self._agent_num,self._action_dim))
        for i in range(self._agent_num):
            for j in range(self._action_dim):
                ac[i] = np.array([1/self._agent_num,1/(2*self._agent_num),1/(2*self._agent_num)])
        return ac, np.zeros((self._agent_num,self._action_dim)),None

    def get_actions(self, observations,current_iteration,path_length,deterministic=False):
        observations = torch.FloatTensor(observations).to(device)
        actions = []
        pis = []
        for i in range(self._agent_num):
            mean = self.actors[i](observations[i])

            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            normal_dist = torch.distributions.MultivariateNormal(mean,cov_mat)
            if deterministic:
                exploration_std = self.get_exploration_std(current_iteration=current_iteration,
                                                           total_iterations=path_length)
                noise = torch.randn_like(mean) * exploration_std
                action = mean + noise
                # Ensure actions are within legal limits
                action = torch.clamp(action, min=0, max=1.0)
                actions.append(action.tolist())

            else:
                action = normal_dist.sample()
                action = torch.clamp(action, min=0, max=1.0)
                actions.append(action[0].tolist())
            if not deterministic:
                log_probs = normal_dist.log_prob(action)
                pis.append(log_probs.tolist())
            else:
                log_probs = None
                pis.append(action.tolist())

        return actions, pis, {}



    def get_exploratory_actions(self, i, observations, exploration_std=0.01):
        observations = torch.tensor(observations).float()
        # Get the mean value of the action of agent i.
        mean= self.actors[i](observations)
        # Generate noise of the same shape as mean and multiply by exploration_std
        noise = torch.randn_like(mean) * exploration_std
        # Add noise to the mean to get exploratory actions
        action = mean + noise
        # Ensure action is within legal limits (if required)
        action = torch.clamp(action, min=0, max=1.0)

        return action

    def train(self):
        self._train(self._env, self, self._pool, self._initial_exploration_done)

    def compute_QValue(self,state,action):
        Qs = []
        batch_size = int(state.size(0))
        for i,critic in enumerate(self.critics):
            agent_state = state[:,i*self._state_dim:(i+1)*self._state_dim]
            agent_action = action[:,i*self._action_dim:(i+1)*self._action_dim]
            agent_id = (torch.ones(batch_size) * i).view(-1, 1)
            agent_input = torch.cat([agent_id,agent_state.type(torch.float32), agent_action.type(torch.float32)], dim=-1)
            Qs.append(critic(agent_input))
        Qs = torch.cat(Qs, dim=1)
        return Qs

    # Train agents to act (Actor) and comment (Critic)
    def _do_training(self, iteration, batch,normal=False):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer
        action, observation, pis, reward, done,next_obs = self._get_memory_dict(iteration, batch)
        actions = [(a.flatten()).tolist() for a in action]

        actions = torch.tensor(actions)
        observations = [torch.tensor(obs).detach() for obs in observation]
        next_obs = [torch.tensor(obs).detach() for obs in next_obs]


        for i in range(self._agent_num):
            # Calculate the advantage function and Q
            advantages, Qs = self.compute_cb_return(observations, actions, i)
            Q = Qs[:,i]
            advantage = advantages.sum(dim=1)

            if normal:
                # Greedy policy: use the current policy network to select the optimal action
                with torch.no_grad():
                    log_pi = self.get_exploratory_actions(i, observation[:, i])
            else:
                log_pi = torch.tensor(pis[:,i])
            log_pi = log_pi.mean(dim=1, keepdim=True)

            actor_loss = -torch.mean(advantage * log_pi)

            # Optimise the Actor network
            actor_optimizer[i].zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)

            # Calculate TD(0) targets
            Qs_target_next = self.critic_targets[i](self.build_input_critic(i,next_obs,actions)).detach()
            Q_target_next = Qs_target_next.sum(dim=1)
            Q_target = torch.zeros(len(reward[:, i]))

            for t in range(len(reward[:, i]) - 1):
                if done[t][i]:
                    Q_target[t] = reward[:, i][t]/self._scale_reward
                else:
                    Q_target[t] = reward[:, i][t]/self._scale_reward + self._gamma * Q_target_next[t+1]

            # Calculate Critic losses
            critic_loss = torch.mean((Q - Q_target) ** 2)


            # Optimise the Critic network
            critic_optimizer[i].zero_grad(set_to_none=False)
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 5)
            critic_optimizer[i].step()

        if self.count == self.target_update_steps:
            # Updating the Target Critic network
            for critic_target, critic in zip(self.critic_targets, self.critics):
                critic_target.load_state_dict(critic.state_dict())
            self.count = 0
        else:
            self.count += 1





    def compute_cb_return(self, observations, actions, agent_id):
        obs = observations
        batch_size = len(observations)
        observations = torch.cat(observations).view(batch_size, self._state_dim * self._agent_num)
        actions = actions.type(torch.float32)
        # 使用基线策略采样动作
        baseline_action = self.sample_baseline_action(obs,batch_size,agent_id)
        baseline_action_tensor = torch.from_numpy(np.array(baseline_action)).type(torch.float32)
        agent_actions = actions.clone()
        agent_actions[:,agent_id * self._action_dim:(agent_id+1)*self._action_dim] = baseline_action_tensor

        Q = self.compute_QValue(observations,actions)
        baseline = self.compute_QValue(observations,agent_actions)
        cb_return = Q - baseline

        return cb_return,Q

    def sample_baseline_action(self, obs,batch,agent_id):
        # Here you can use a baseline policy to sample actions, such as a randomised strategy or a pre-trained strategy
        obs = torch.stack(obs).float()
        obs = obs[:,agent_id]
        mean = self.actors[agent_id](obs)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        normal_dist = torch.distributions.MultivariateNormal(mean, cov_mat)
        actions = normal_dist.sample()

        return actions

    def compute_log_prob(self, pi, actions):
        actions = torch.tensor(actions)
        alpha = torch.tensor(pi[:, 0])
        beta = torch.tensor(pi[:, 1])
        dist = Beta(alpha, beta)
        log_prob = dist.log_prob(actions)
        return log_prob

    def _get_memory_dict(self, iteration, batch):
        memory_dict = {
            'observations_ph': batch['observations'],
            'actions_ph': batch['actions'],
            'pis_ph': batch['pis'],
            'next_observations_ph': batch['next_observations'],
            'rewards_ph': batch['rewards'],
            'terminals_ph': batch['terminals'],
        }
        if iteration is not None:
            memory_dict['iteration_pl'] = iteration

        return memory_dict['actions_ph'], memory_dict['observations_ph'], memory_dict['pis_ph'], memory_dict[
            'rewards_ph'], \
            memory_dict['terminals_ph'], memory_dict['next_observations_ph']

    # Inputs for building a Critic network
    def build_input_critic(self, agent_id, observations, actions):
        batch_size = len(observations)
        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)
        observations = torch.cat(observations).view(batch_size, self._state_dim * self._agent_num)
        agent_state = observations[:, agent_id * self._state_dim:(agent_id + 1) * self._state_dim]
        agent_action = actions[:, agent_id * self._action_dim:(agent_id + 1) * self._action_dim]
        input_critic = torch.cat([ids,agent_state.type(torch.float32), agent_action.type(torch.float32)], dim=-1)

        return input_critic


    def get_exploration_std(self, initial_std=0.5, final_std=0.01, current_iteration=0, total_iterations=10000):
        # Calculate the current exploration standard deviation
        exploration_std = initial_std - (initial_std - final_std) * (current_iteration / total_iterations)
        # Ensure that the exploratory standard deviation is not less than the final standard deviation
        exploration_std = max(exploration_std, final_std)
        return exploration_std

    def get_snapshot(self, epoch):
        # If Save Complete State is set, save the state of the entire algorithm
        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
                'agent': self.agent,
                'actors': self.actors,
                'critic': self.critics,
                'critic_target': self.critic_targets,
                'env': self._env
            }
        return snapshot

    def __getstate__(self):
        d = Serializable.__getstate__(self)

        # Update the state dictionary, including the value function,, the parameters of the policy, the state of the environment and the pool.
        d.update({
            'critic': [critic.get_parameter() for critic in self.critics],
            'actors': [actor.get_parameter() for actor in self.actors],
            'agents': self.agent,
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

        # Setting value functions, network parameters, environment and pool state
        self.critic.__setstate__(d['critic'])
        for i in range(len(self.actors)):
            self.actors[i].__setstate__(d['actor'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])

    def _init_training(self, env, agent, pool,epoch):
        super(COMA, self)._init_training(env, agent, pool,0)
