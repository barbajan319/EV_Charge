import numpy as np
# from env5 import MultiAgentEVCharge_V5
from envSINGLE import  SingleAgentEVCharge_V5
import torch as T
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import time
import math
import tkinter as tk
import random
import shutil
import torch.nn.utils as nn_utils
from tqdm import tqdm
import matplotlib.pyplot as plt




T.autograd.set_detect_anomaly(True)
# Define the directory path for the new TensorBoard logs
new_log_dir = 'C:/Users/luisb/EVCHARGING/tmp/maddpg/new_logs'

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_counter = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        #State Memory
        self.state_memory = np.zeros((self.mem_size, critic_dims)) 
        self.new_state_memory = np.zeros((self.mem_size, critic_dims)) 
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        #For critic value for the terminal state is always 0, when an agent reaches a terminal state
        #the episode terminates and no future rewards follow
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        
        self.init_actor_memory()
        
    def init_actor_memory(self):
        """
        Initialize memory for actor states and actions for each agent.
        """
        # Action memory consists of the action taken by each agent in that particular time step  
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        
        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))
    
    def store_transition(self, raw_obs, state, action, rewards, new_raw_obs, new_state, termination):
        """
        Function to store transitions in memory
        """
        index = self.mem_counter % self.mem_size
        #Iterate over all agents getting observations and storing them to the appropriate agent's actor
        for agent_idx in range(self.n_agents):
            state_vec = list(raw_obs["EV_%s" % agent_idx].values())
            state_vec_ = list(new_raw_obs["EV_%s" % agent_idx].values())
            vector = np.concatenate([arr.flatten() for arr in state_vec])
            vector_ = np.concatenate([arr.flatten() for arr in state_vec_])
            
            self.actor_state_memory[agent_idx][index] = vector
            self.actor_new_state_memory[agent_idx][index] = vector_

            self.actor_action_memory[agent_idx][index] = action[agent_idx]
            
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = [reward for reward in rewards.values()]
        self.terminal_memory[index] = termination
        self.mem_counter += 1
    
    def sample_buffer(self):
        #Indicate highest position filled in memory to avoid sampling zeros
        max_mem = min(self.mem_counter, self.mem_size)
        
        batch = np.random.choice(max_mem, self.batch_size, replace = False) #False to avoid getting same memory twice
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        actor_states = []
        actor_new_states = []
        actions = []
        
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
      
        return actor_states, states, actions, rewards, actor_new_states, new_states, terminal
    
    def ready(self):
        if self.mem_counter >= self.batch_size:
            return True
        return False
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, 
                chkpt_dir):
        super(CriticNetwork, self).__init__()
        
        self.chkpt_file = os.path.join(chkpt_dir, name)
        
        #First layer determined by combination of environment's state and actions taken by agent
        #Critic takes a full state observation vector of the whole system and full actions for each agent
        self.fc1 = nn.Linear(input_dims+n_agents*(n_actions) , fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 64)
        self.fc4 = nn.Linear(64, 32)
        #output Q value
        self.q = nn.Linear(64, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        #Optional GPU Usage:
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def  forward(self, state, action):  
        X = F.relu(self.fc1(T.cat((state, action), dim=1)))
        X = F.relu(self.fc2(X))
        # X = F.relu(self.fc3(X))
        # X = F.relu(self.fc4(X))
        q = self.q(X)

        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, 
               chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        #first layer takes only each agents observations
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 64)
        self.fc4 = nn.Linear(64, 32)
        self.continous_output = nn.Linear(64, n_actions)
    
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #Optional GPU Usage:
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        cont_out = self.continous_output(x)
    
        charge_rate = T.tanh(cont_out)

        return charge_rate
    def save_checkpoint(self):  
        T.save(self.state_dict(), self.chkpt_file)
        
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
        
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir, 
              alpha, beta, fc1, fc2, gamma, tau):
  
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = "EV_%s" % agent_idx
        self.n_agents = n_agents
        self.epsilon = 1 
        self.epsilon = T.tensor(self.epsilon, dtype=T.float32)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        name = self.agent_name+"_actor", chkpt_dir=chkpt_dir)
        
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, 
                                    name = self.agent_name+"_critic", chkpt_dir=chkpt_dir)

        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        name = self.agent_name+"_target_actor", chkpt_dir=chkpt_dir)
        
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, 
                                    name = self.agent_name+"_target_critic", chkpt_dir=chkpt_dir)
        
        self.update_network_parameters(tau = 1)
  
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
      
        # Get the named parameters of the target actor network and the actor network
        # Named parameters are pairs of parameter names and parameter tensors (Weights and biases).
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        # Create dictionaries to store the state dictionaries of the target actor and actor
        # Each dictionary maps parameter names to parameter tensors.
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        actor_state_dict = {k: v for k, v in actor_state_dict.items() if 'bn' not in k}
    # Perform soft parameter updates by blending the actor network's parameters into the target actor network
        for name in actor_state_dict:
            # Blend the current actor's parameter with the corresponding target actor's parameter
            # using a smoothing factor (tau) to control the degree of blending
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict, strict=False)
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict, strict = False)
    
    def choose_action(self, observation):
        soc = T.tensor(observation['SOC']/450, dtype = T.float32)
        target_soc = T.tensor(observation['Target_SOC']/450, dtype=T.float32)
        time_left_depot = T.tensor(observation['Time_left_Depot']/96, dtype=T.float32)
        #available_stations = T.tensor(observation['Available_Stations'], dtype=T.float32)
        # num_zeros = self.n_agents - available_stations.shape[0]
        # available_stations = T.cat([available_stations, T.zeros((num_zeros))])
        #state = T.cat((soc, target_soc, time_left_depot, current_energy_price, future_energy_window, available_stations,
        #               load_on_transformer), dim = -1)
        state = T.cat((soc, target_soc, time_left_depot), dim = -1)
        state = state.to(self.actor.device)
        charge_rate = self.actor.forward(state)
  
        if random.random() < self.epsilon:
            charge_rate = self.explore_with_noise(charge_rate)
      
        return  charge_rate.detach().numpy()[0]
  
    def explore_with_noise(self, action):
        noise = T.normal(-1,1, size = action.shape)
        noisy_action = action + self.epsilon * noise
        return T.clamp(noisy_action, -1.0, 1.0)
  
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

  
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        
        
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
               env, alpha, beta, fc1, fc2, gamma, tau, chkpt_dir = 'tmp/maddpg/'):
        self.writer = SummaryWriter()
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.fc1 = fc1
        self.fc2 = fc2
        self.gamma = gamma
        self.tau = tau
    
        for agent_idx in range(self.n_agents):

            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                               alpha, beta, fc1, fc2, gamma, tau))
      # self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions_1, n_actions_2, agent_idx,
      #                          alpha=alpha, beta=beta, chkpt_dir=chkpt_dir))
      
    def save_checkpoint(self, agent):
        print('....saving checkpoint.....', agent.agent_name)
        agent.save_models()
      
    def load_checkpoint(self):
        print('....loading checkpoint.....')
        for agent in self.agents:
            agent.load_models()
      
    def choose_action(self, obs):
        actions = [] #might need to update this to a dictionary
        for agent_idx, agent in enumerate(self.agents):
            actions.append(agent.choose_action(obs["EV_%s" % agent_idx]))
        return actions
  
    def update_epsilon(self):
        for agent in self.agents:
            pass
        agent.update_epsilon()
  
    def learn(self, memory, episode):
        if not memory.ready():
            return
        actor_states, states, actions, rewards, actor_new_states, states_, terminations = memory.sample_buffer()
        device = self.agents[0].actor.device
        states = T.tensor(states, dtype=T.float32).to(device)
        actions = T.tensor(actions_np, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype = T.float32).to(device)
        states_ = T.tensor(states_, dtype = T.float32).to(device)
        terminations = T.tensor(terminations).to(device)
        
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
    
        for agent_idx, agent in enumerate(self.agents):
            #estimate action values for the next state according to actor network
            new_states = T.tensor(actor_new_states[agent_idx], dtype = T.float32).to(device)
            new_charge_rate = agent.target_actor.forward(new_states)
            all_agents_new_actions.append((new_charge_rate))
            
            #Action for current state from actor network
            mu_states = T.tensor(actor_states[agent_idx], dtype = T.float32).to(device)
            charge_rate= agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append((charge_rate))

      
      #actions agent actually took
     
            old_agents_actions.append(actions[agent_idx])  
      
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim = 1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim = 1)
        old_actions =T.cat([acts for acts in old_agents_actions], dim = 1)
    
        for agent_idx, agent in enumerate(self.agents):
            #get the states and new action for the target critic network and flatten them.
            #critic values with target critic
            #One-step lookahead TD-error:
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            #ensure that terminal states are not include in future rewards
            critic_value_[terminations[:,0]] = 0.0
            #critic values using the local critic 
            # network, how good the action actually was
            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            mean_rewards = T.mean(rewards[:, agent_idx])
            std_rewards = T.std(rewards[:, agent_idx])
            normalized_rewards = (rewards[:, agent_idx] - mean_rewards) / (std_rewards + 1e-8)

            #target = normalized_rewards + agent.gamma*critic_value_
            target = rewards[:, agent_idx] + agent.gamma*critic_value_
            # target = target.clone().detach().requires_grad_(True)
            # critic_value = critic_value.clone().detach().requires_grad_(True)
            #calculate the loss of the current critic value 
            critic_loss = F.mse_loss(target, critic_value)
      
            # print("critic_loss: ", critic_loss)
            self.writer.add_scalar(f"EV_{agent_idx}/Loss/Critic", critic_loss, episode)
        
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph = True)
            # nn_utils.clip_grad_norm_(agent.critic.parameters(), max_norm =1)
            agent.critic.optimizer.step()

        
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)#.requires_grad_(True)
            # actor_loss = actor_loss.clone().detach().requires_grad_(True)
            self.writer.add_scalar(f"EV_{agent_idx}/Loss/Actor", actor_loss, episode)
        
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph = True)
            # nn_utils.clip_grad_norm_(agent.critic.parameters(), max_norm =1)
            agent.actor.optimizer.step()
            
            agent.update_network_parameters()
        
def obs_dict_to_state_vector(data, n_agents):
    state = np.array([])
    for agent in data.keys():
        for obs in data[agent]:
            state = np.append(state, data[agent][obs])
    return state



if __name__  == "__main__":

    env = SingleAgentEVCharge_V5(num_agents = 1)
    obs = env.reset()
    agent_names = env.agents
    n_agents = env.num_agents
    n_stations = env.num_stations
    obs = env.observation_space(env.agents[0])
    actor_dims = [3]
    overcharge_times = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
    overcharge_mean = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
    charging_rate_ = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
    undercharge = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
    complete_charge = []
    charge_att = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
    gap_time = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
  #in order to manage varying number of agents and stations, we need to make the dimensions for the largest observation space possible which is num stations = num agents
#   dims = sum(obs[key].shape[0] if len(obs[key].shape) > 0 else 1 for key in obs.keys())
#   for i in range(n_agents):
#     actor_dims.append(dims)
    
    critic_dims = sum(actor_dims)
    acts = env.action_space(env.agents[0])
    n_actions = 1
    maddppg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, env,
                          alpha=0.001, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau = 0.01, chkpt_dir='C:/Users/luisb/EVCHARGING/tmp/maddpg_v5')
  
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=2048)
  
  
    PRINT_INTERVAL = 500
    N_EPS = 100000
    total_steps = 0
    score_history = {'EV_0': [], 'EV_1': [], 'EV_2': [], 'EV_3': []}
    evaluate = False
    best_score = {'EV_0': -math.inf, 'EV_1': -math.inf, 'EV_2': -math.inf, 'EV_3': -math.inf}
    truncations = False
    avg_score = {'EV_0': 0, 'EV_1': 0, 'EV_2': 0, 'EV_3': 0}
    task_completion = []
    indv_completion = {'EV_0': 0, 'EV_1': 0, 'EV_2': 0, 'EV_3': 0}
    if evaluate:
        maddppg_agents.load_checkpoint()
    for i in range(N_EPS):
        obs = env.reset()
        score = {'EV_0': 0, 'EV_1': 0, 'EV_2': 0, 'EV_3': 0}
        terminations = env.terminations
        truncations = env.truncations
        episode_step = 0
        while not all(terminations.values()) and not truncations:
            # if evaluate:
            #     # env.render()
            actions = maddppg_agents.choose_action(obs)
            obs_, rewards, terminations,truncations, _, tc = env.step(actions)

      
            state = obs_dict_to_state_vector(obs, n_agents)
            state_ = obs_dict_to_state_vector(obs_, n_agents)
     
            terminations_list = list(terminations.values())
            memory.store_transition(obs, state, actions, rewards, obs_, state_, terminations_list)
            obs = obs_
      
            total_steps  += 1
            episode_step += 1


        for agent in rewards.keys():
            score[agent] = rewards[agent]
            score_history[agent].append(score[agent])
            avg_score[agent] = np.mean(score_history[agent][-100:])

        maddppg_agents.update_epsilon()
        if not evaluate:
            for agent in rewards.keys():
                maddppg_agents.writer.add_scalar(f"{agent}/Reward_Episode", rewards[agent], i)
            if memory.ready() and i % 100 == 0:
                maddppg_agents.learn(memory, i)
                for agent in maddppg_agents.agents:
                    if avg_score[agent.agent_name] > best_score[agent.agent_name]:
                        maddppg_agents.save_checkpoint(agent)
                        best_score[agent.agent_name] = avg_score[agent.agent_name]
                        
        

        
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f"Episode :, {i}, Average Score: {avg_score} Best Score: {best_score}")
            print(f"Rewards:  {rewards}\n")

        if evaluate:
            task_completion.append(tc)
            indv_tsk = env.indv_task
            for agent in agent_names:
                if indv_tsk[agent]:
                    indv_completion[agent] += 1
                    
        #Get Metrics
        for EV in agent_names:
          # Times agents overcharged their battery in an episode
          overcharge_times[EV].append(env.contributions[EV]["overcharge_bat"][0])
          # The average kWh the EV overcharged
          
          avg = sum(env.contributions[EV]["overcharge_bat"][1]) / env.contributions[EV]["overcharge_bat"][0] if env.contributions[EV]["overcharge_bat"][0] != 0 else 0
          overcharge_mean[EV].append(avg)
          # Times EV attempted to charge while it had already reached target SOC
          avg_rate = env.contributions[EV]["charging_rate"][1] / env.contributions[EV]["charging_rate"][0]
          charging_rate_[EV].append(avg_rate)
          #kWh short from completing goal
          undercharge[EV].append(env.contributions[EV]['undercharge'][1])
          # How many times each EV entered the charging station
          charge_att[EV].append(env.charging_attempt[EV])
          #Timesteps taken to close the gap
          gap_time[EV].append(env.contributions[EV]['gap'][0])
        #How many Evs completed charging
        complete_charge.append(env.complete_charge)
# Create two subplots: one for overcharge_times and one for overcharge_att
# Plot data from overcharge_times

    for agent in maddppg_agents.agents:
        maddppg_agents.save_checkpoint(agent)
        
    plt.figure(figsize=(6, 4))
    for ev, values in overcharge_times.items():

      plt.plot(values, label=ev, marker='o')
    plt.title('Overcharge Times')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Plot data from overcharge_mean
    plt.figure(figsize=(6, 4))
    for ev, values in overcharge_mean.items():
      plt.plot(values, label=ev, marker='o')
    plt.title('Overcharge Mean')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()



# Plot data from undercharge
    plt.figure(figsize=(6, 4))
    for ev, values in undercharge.items():
      plt.plot(values, label=ev, marker='o')
    plt.title('Undercharge')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Plot data from charge_att
    plt.figure(figsize=(6, 4))
    for ev, values in charging_rate_.items():
      plt.plot(values, label=ev, marker='o')
    plt.title('Charge Rate')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Plot data from gap_time
    plt.figure(figsize=(6, 4))
    for ev, values in gap_time.items():
      plt.plot(values, label=ev, marker='o')
    plt.title('Gap Time')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Plot data from complete_task
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(complete_charge)), complete_charge)
    plt.title('Complete Task')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.show()
