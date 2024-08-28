import traceback
import sys
try:
    import numpy as np
    from EVenv import MultiAgentEVCharge
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
    
    T.autograd.set_detect_anomaly(True)

    class MultiAgentReplayBuffer:
        def __init__(self, max_size, critic_dims, actor_dims, n_actions_1, n_actions_2, n_agents, batch_size):
            self.mem_size = max_size
            self.mem_counter = 0
            self.n_agents = n_agents
            self.batch_size = batch_size
            self.n_actions = n_actions_1 + n_actions_2
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
                
                action_1, action_2 = action[agent_idx]

                # if np.isnan(action_1).any():
                #     print("Continous action contains NaN values: ", action_1, agent_idx)
                # if T.isnan(action_2).any():
                #     print("Discrete action contains NaN values: ", action_2) 
            
                action_1 = np.array(action_1)
                action_2 = action_2.detach().numpy()    
                actions = np.concatenate([action_1, action_2])
                
                if len(vector)  != self.actor_dims[agent_idx]:
                    zero_pad = self.actor_dims[0] -  len(vector)
                    vector = np.pad(vector, (0, zero_pad), mode='constant')
                    vector_ = np.pad(vector_, (0, zero_pad), mode='constant')
            
                self.actor_state_memory[agent_idx][index] = vector
                self.actor_new_state_memory[agent_idx][index] = vector_
                self.actor_action_memory[agent_idx][index] = actions
        
            self.state_memory[index] = state
            self.new_state_memory[index] = new_state
            self.reward_memory[index] = [reward for reward in rewards.values()]
            self.terminal_memory[index] = termination
            self.mem_counter += 1
        
  #Function to sample buffer
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
        def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions_1, n_actions_2, name, 
               chkpt_dir):
            super(CriticNetwork, self).__init__()
    
            self.chkpt_file = os.path.join(chkpt_dir, name)
            
            #First layer determined by combination of environment's state and actions taken by agent
            #Critic takes a full state observation vector of the whole system and full actions for each agent
            self.fc1 = nn.Linear(input_dims+n_agents*(n_actions_1 + n_actions_2) , fc1_dims)
            self.fc2 = nn.Linear(fc1_dims, fc2_dims)
            self.fc3 = nn.Linear(fc2_dims, 64)
            # self.double()
            #output Q value
            self.q = nn.Linear(64, 1)
            
            self.optimizer_critic = optim.Adam(self.parameters(), lr=beta)

            #Optional GPU Usage:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            self.to(self.device)
    
        def  forward(self, state, action):  
            X = F.relu(self.fc1(T.cat((state, action), dim=1)))
            X = F.relu(self.fc2(X))
            X = F.relu(self.fc3(X))
            # X = F.relu(self.fc3(X))
            q = self.q(X)
            return q
  
        def save_checkpoint(self):
            T.save(self.state_dict(), self.chkpt_file)
    
        def load_checkpoint(self):
            self.load_state_dict(T.load(self.chkpt_file))
            
    class ActorNetwork(nn.Module):
        def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions_1, n_actions_2, name, 
               chkpt_dir):
            super(ActorNetwork, self).__init__()
            self.chkpt_file = os.path.join(chkpt_dir, name)
            #first layer takes only each agents observations
            self.fc1 = nn.Linear(input_dims, fc1_dims)
            self.fc2 = nn.Linear(fc1_dims, fc2_dims)
            self.relu = nn.ReLU(inplace = False)
            self.fc3 = nn.Linear(fc2_dims, 64)
            # self.double()
            
            self.continous_output = nn.Linear(64, n_actions_1)
            
            self.discrete_output = nn.Linear(64, n_actions_2)
            
            self.optimizer_actor = optim.Adam(self.parameters(), lr=alpha)
            #Optional GPU Usage:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            self.to(self.device)
    
        def forward(self, state):
            x = self.fc1(state)
            x = self.relu(x)
            if T.isnan(x).any():
                print("x1 is NaN")
            x = self.fc2(x)
            x = self.relu(x)
            if T.isnan(x).any():
                print("x2 is NaN")
            x = self.fc3(x)
            x = self.relu(x)
            if T.isnan(x).any():
                print("x3 is NaN")
            # x = F.relu(self.fc3(x))
            # if T.isnan(x).any():
            #   print("x is NaN")
            cont_out = self.continous_output(x)
            dis_out = self.discrete_output(x)
            
            charge_rate = T.sigmoid(cont_out)
            
            charge_decision = T.softmax(dis_out, dim=-1)    

            return charge_rate, charge_decision
      
        def save_checkpoint(self):
            T.save(self.state_dict(), self.chkpt_file)
    
        def load_checkpoint(self):
            self.load_state_dict(T.load(self.chkpt_file))   
            
    class Agent:
        def __init__(self, actor_dims, critic_dims, n_actions_1, n_actions_2, n_agents, agent_idx, chkpt_dir, 
              alpha, beta, fc1, fc2, gamma, tau):
  
            self.gamma = gamma
            self.tau = tau
            self.n_actions_1 = n_actions_1
            self.n_actions_2 = n_actions_2 
            self.agent_name = "EV_%s" % agent_idx
            self.n_agents = n_agents

            self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions_1, n_actions_2,
                                            name = self.agent_name+"_actor", chkpt_dir=chkpt_dir)
            
            self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions_1, n_actions_2, 
                                        name = self.agent_name+"_critic", chkpt_dir=chkpt_dir)

            self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions_1, n_actions_2,
                                            name = self.agent_name+"_target_actor", chkpt_dir=chkpt_dir)
            
            self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions_1, n_actions_2, 
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

            # Perform soft parameter updates by blending the actor network's parameters into the target actor network
            for name in actor_state_dict:
                # Blend the current actor's parameter with the corresponding target actor's parameter
                # using a smoothing factor (tau) to control the degree of blending
                actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

            self.target_actor.load_state_dict(actor_state_dict)

            target_critic_params = self.target_critic.named_parameters()
            critic_params = self.critic.named_parameters()

            target_critic_state_dict = dict(target_critic_params)
            critic_state_dict = dict(critic_params)

            for name in critic_state_dict:
                critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()
            self.target_critic.load_state_dict(critic_state_dict)
  
        def choose_action(self, observation):
            soc = T.tensor(observation['SOC']/450, dtype = T.float32)
            target_soc = T.tensor(observation['Target_SOC']/450, dtype=T.float32)
            time_left_depot = T.tensor(observation['Time_left_Depot']/96, dtype=T.float32)
            current_energy_price = T.tensor(observation['Current_Energy_Price'], dtype=T.float32)
            future_energy_window = T.tensor(observation['Future_Energy_Window'], dtype=T.float32)
            load_on_transformer = T.tensor(observation['Load_on_transformer']/1350, dtype=T.float32)
            available_stations = T.tensor(observation['Available_Stations'], dtype=T.float32)
            num_zeros = self.n_agents - available_stations.shape[0]
            available_stations = T.cat([available_stations, T.zeros((num_zeros))])
            
            state = T.cat((soc, target_soc, time_left_depot, current_energy_price, future_energy_window, available_stations,
                            load_on_transformer), dim = -1)
            state = state.to(self.actor.device)
            charge_rate, charge_decision = self.actor.forward(state)
            
            epsilon = 0.1
            noise_level = 0.5
        
            # charge_rate = ((charge_rate)+1)/ 2
            charge_rate = T.clamp(charge_rate, 0, 1)
            # Choose action based on the probability distribution
        
            if random.random() < epsilon:
                noisy_logits = T.log(charge_decision) + T.randn_like(charge_decision) * noise_level
                charge_decision = T.softmax(noisy_logits, dim=-1)
                noise = T.rand(self.n_actions_1)
                charge_rate = charge_rate + noise 
            
            return  charge_rate.detach().numpy(), charge_decision
  
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
        def __init__(self, actor_dims, critic_dims, n_agents, n_actions_1, n_actions_2,
                    env, alpha, beta, fc1, fc2, gamma, tau, chkpt_dir = 'tmp/maddpg/'):
            self.writer = SummaryWriter()
            self.agents = []
            self.n_agents = n_agents
            self.n_actions_1 = n_actions_1
            self.n_actions_1 - n_actions_2
            self.alpha = alpha
            self.beta = beta
            self.fc1 = fc1
            self.fc2 = fc2
            self.gamma = gamma
            self.tau = tau
            
            for agent_idx in range(self.n_agents):

                self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions_1, n_actions_2, n_agents, agent_idx, chkpt_dir,
                                    alpha, beta, fc1, fc2, gamma, tau))
            # self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions_1, n_actions_2, agent_idx,
            #                          alpha=alpha, beta=beta, chkpt_dir=chkpt_dir))
            
        def save_checkpoint(self):
            print('....saving checkpoint.....')
            for agent in self.agents:
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
        
        def learn(self, memory, episode):
            if not memory.ready():
                return
            actor_states, states, actions, rewards, actor_new_states, states_, terminations = memory.sample_buffer()

            device = self.agents[0].actor.device
            # print("terms", terminations)
            states = T.tensor(states, dtype=T.float32).to(device)
            actions_np = np.array(actions, dtype=np.float32)  # Convert the list of NumPy arrays to a single NumPy array
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
                new_charge_rate, new_charge_decision = agent.target_actor.forward(new_states)
                all_agents_new_actions.append((new_charge_rate))
                all_agents_new_actions.append((new_charge_decision))
                
                #Action for current state from actor network
                mu_states = T.tensor(actor_states[agent_idx], dtype = T.float32).to(device)
                charge_rate, charge_decision = agent.actor.forward(mu_states)
                all_agents_new_mu_actions.append((charge_rate))
                all_agents_new_mu_actions.append((charge_decision))
                
                #actions agent actually took
                
                old_agents_actions.append(actions[agent_idx])  
                
            new_actions = T.cat([acts for acts in all_agents_new_actions], dim = 1)
            mu = T.cat([acts for acts in all_agents_new_mu_actions], dim = 1)
            old_actions =T.cat([acts for acts in old_agents_actions], dim = 1)
            
            #Cost functions
            for agent_idx, agent in enumerate(self.agents):
                try:
                    #get the states and new action for the target critic network and flatten them.
                    #critic values with target critic
                    #One-step lookahead TD-error:
                    critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                    # print(f"crtic_value_: {critic_value_}")
                    # print(f"terminations:", [terminations])
                    #ensure that terminal states are not include in future rewards
                    critic_value_[terminations[:,0]] = 0.0
                    # print(f"crtic_value_2: {critic_value_}")
                    #critic values using the local critic 
                    # network, how good the action actually was
                    critic_value = agent.critic.forward(states, old_actions).flatten()
                    
                    mean_rewards = T.mean(rewards[:, agent_idx])
                    std_rewards = T.std(rewards[:, agent_idx])
                    normalized_rewards = (rewards[:, agent_idx] - mean_rewards) / (std_rewards + 1e-8)

                    # print(f"crtic_value: {numpy_array}")
                    target = normalized_rewards + agent.gamma*critic_value_
                    #print("target: ", target)
                    #target = rewards[:, agent_idx] + agent.gamma*critic_value_
                    #print("critic value: ", critic_value)
                    #calculate the loss of the current critic value 
                    critic_loss = F.mse_loss(target, critic_value)
                    # print("critic_loss: ", critic_loss)
                    self.writer.add_scalar(f"EV_{agent_idx}/Loss/Critic", critic_loss, episode)
                    
                    agent.critic.optimizer_critic.zero_grad()
                    critic_loss.backward(retain_graph = True)
                    agent.critic.optimizer_critic.step()
                        
                    actor_loss = agent.critic.forward(states, mu).flatten()
                    # print("actor_loss", actor_loss)
                    actor_loss = -T.mean(actor_loss)
                    self.writer.add_scalar(f"EV_{agent_idx}/Loss/Actor", actor_loss, episode)
                    agent.actor.optimizer_actor.zero_grad()
                    actor_loss.backward(retain_graph = True)
                    agent.actor.optimizer_actor.step()
                    
                    agent.update_network_parameters()   
                except:
                    traceback.print_exc(file=sys.stdout)
                    print("Error occurred in agent:", agent_idx)  
                
    def obs_dict_to_state_vector(data, n_agents):
        state = np.array([])
        for agent in data.keys():
            for obs in data[agent]:
                if obs == 'Available_Stations':
                    pad = n_agents - len(data[agent][obs])
                    x = np.pad(data[agent][obs], (0, pad), 'constant')
                    state = np.append(state, x)
                else:
                    state = np.append(state, data[agent][obs])
        return state   
    
    if __name__  == "__main__":
        env = MultiAgentEVCharge(num_agents = 4)
        env.reset()
        n_agents = env.num_agents
        n_stations = env.num_stations
        obs = env.observation_space(env.agents[0])
        actor_dims = []
        #in order to manage varying number of agents and stations, we need to make the dimensions for the largest observation space possible which is num stations = num agents
        dims = sum(n_agents if key == "Available_Stations" else obs[key].shape[0] if len(obs[key].shape) > 0 else 1 for key in obs.keys())
        for i in range(n_agents):
            actor_dims.append(dims)
            
        critic_dims = sum(actor_dims)
        acts = env.action_space(env.agents[0])
        n_actions_1 = 1
        n_actions_2 = acts.spaces[1].n
  
        maddppg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions_1, n_actions_2, env,
                          alpha=0.001, beta=0.01, fc1=64, fc2=128, gamma=0.99, tau = 0.01, chkpt_dir='C:/Users/luisb/EVCHARGING/tmp/maddpg')
  
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions_1, 
                                  n_actions_2, n_agents, batch_size=1024)
  
        PRINT_INTERVAL = 500
        N_EPS = 10000
        total_steps = 0
        score_history = []
        evaluate = False
        best_score = -math.inf
        truncations = False
        # maddppg_agents.load_checkpoint()
        if evaluate:
            maddppg_agents.load_checkpoint()
        for i in range(N_EPS):
            obs, infos = env.reset()
            score = 0
            terminations = env.terminations
            truncations = env.truncations
            episode_step = 0
            while not all(terminations.values()) and not truncations:
                if evaluate:
                    env.render()
                
                actions = maddppg_agents.choose_action(obs)
                obs_, rewards, terminations,truncations, _ = env.step(actions)
                # print(truncations, env.timestep)
                state = obs_dict_to_state_vector(obs, n_agents)
                state_ = obs_dict_to_state_vector(obs_, n_agents)
      
                if np.isnan(state).any():
                    print("State tensor contains NaN values:")
                    print(state)
      
                for cont_action, dis_action in actions:
                    # Now you can use cont_action and dis_action for each tuple
                    if np.isnan(cont_action).any():
                        print("Continous action contains NaN values: ", cont_action)
                    if T.isnan(dis_action).any():
                        print("Discrete action contains NaN values: ", dis_action)

                memory.store_transition(obs, state, actions, rewards, obs_, state_, terminations)
      
      #for agent in rewards.keys():
        
                obs = obs_
                score = sum(rewards.values())
                total_steps  += 1
                episode_step += 1
      
            if not evaluate:
                maddppg_agents.learn(memory, i)
            for agent in rewards.keys():
                maddppg_agents.writer.add_scalar(f"{agent}/Reward_Episode", rewards[agent], i)
                # maddppg_agents.writer.add_scalar("Rewards/Episode", score, i)
      
            score_history.append(score)
            avg_score = np.mean(score_history[-50:])
    
            if not evaluate:
                if avg_score > best_score:
                    maddppg_agents.save_checkpoint()
                    best_score = avg_score
            if i % PRINT_INTERVAL == 0 and i > 0:
                print(f"Episode i:, {i}, Average Score: {avg_score} Best Score: {best_score}")    
            
except Exception as e:
    traceback.print_exc()
