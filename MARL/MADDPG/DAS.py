import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from envSINGLE import SingleAgentEVCharge_V5
import copy
import matplotlib.pyplot as plt
T.autograd.set_detect_anomaly(True)

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))


    def store_transition(self, raw_obs, state, action, reward, 
                               raw_obs_, state_, done):
  
        index = self.mem_cntr % self.mem_size
        for agent_idx in range(self.n_agents):
            raw = np.array(list(raw_obs[agent_idx].values()))
            raw_ = np.array(list(raw_obs_[agent_idx].values()))
            _raw_obs = [value[0] for value in raw] 
            _raw_obs_ = [value[0] for value in raw_] 
            self.actor_state_memory[agent_idx][index] = _raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = _raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        rewards = [r for r in reward.values()]
        self.reward_memory[index] = rewards
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.tanh(self.pi(x))
        return pi.clone()

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))       
        
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')

        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        obs = []
        observation_values = np.array(list(observation.values()))
        for value in observation_values:
            obs.append(value[0])
        state = T.tensor(obs, dtype=T.float32).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

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
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))
            
  



    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device
        actions = np.array(actions)
        states = T.tensor(states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype = T.float32).to(device)
        states_ = T.tensor(states_, dtype=T.float32).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float32).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float32).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            mean_rewards = T.mean(rewards[:, agent_idx])
            std_rewards = T.std(rewards[:, agent_idx])
            normalized_rewards = (rewards[:, agent_idx] - mean_rewards) / (std_rewards + 1e-8)
            
            target = normalized_rewards + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target.detach(), critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            

            agent.update_network_parameters()
        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.step()
            agent.critic.optimizer.step()

def obs_list_to_state_vector(observation):
    state = np.array([])
    for outer, inner, in observation.items():
        for inner_key, value in inner.items():
            state = np.concatenate([state, value])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    def generate_evs(num_evs, flag):
        ev_dict = {}
        for i in range(num_evs):
            if flag  == 1:
                ev_dict[i] = []
            elif flag == 2:
                ev_dict[i] = -math.inf
            
            elif flag == 3:
                ev_dict[i] = 0
        return ev_dict
    
    env = SingleAgentEVCharge_V5(num_agents=2, alpha=10, beta=0.01, gamma=200)
    alpha = env.alpha
    beta = env.beta
    gamma = env.gamma
    n_agents = env.num_agents
    overcharge_times = generate_evs(n_agents, 1)
    overcharge_mean = generate_evs(n_agents,1)
    charging_rate_ = generate_evs(n_agents,1)
    undercharge = generate_evs(n_agents,1)
    complete_charge = []
    charge_att = generate_evs(n_agents,1)
    gap_time = generate_evs(n_agents,1)
    possible_agents = env.possible_agents
    actor_dims = [3,3]
    critic_dims = sum(actor_dims)
    

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 1


    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                        alpha=0.01, beta=0.01,
                            chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                       n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 25000
    # MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    task_completion = []
    indv_completion = generate_evs(n_agents,3)

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        terminations = env.terminations
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not all(terminations.values()):
            if evaluate:
                env.render()
                 #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            

            obs_, reward, done, truncations, info, task = env.step(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
               
            obs = obs_
            score += sum(reward)
            total_steps += 1
            episode_step += 1
            

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                
            if i % 10 == 0:
               maddpg_agents.learn(memory)
               
        if i % PRINT_INTERVAL == 0 and i > 0:
           print('episode', i, 'average score {:.1f}'.format(avg_score))
           
        #Get Metrics
        for EV in possible_agents:
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

    folder_name = f'n_agents{n_agents}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    window_size = 100
    plt.figure(figsize=(6, 4))
    for ev, values in overcharge_times.items():
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(values)), moving_avg, label=ev)
    plt.title(f'Overcharge Times [a={alpha}, B={beta}, y={gamma}]')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Overcharge_times_{alpha}_{beta}_{gamma}.png'))
    plt.close()

    # Plot data from overcharge_mean
    plt.figure(figsize=(6, 4))
    for ev, values in overcharge_mean.items():
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(values)), moving_avg, label=ev)
    plt.title(f'Overcharge Mean [a={alpha}, B={beta}, y={gamma}]')
    plt.xlabel('Index')
    plt.ylabel('kW')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Overcharge_mean_{alpha}_{beta}_{gamma}.png'))
    plt.close()



    # Plot data from undercharge
    plt.figure(figsize=(6, 4))
    for ev, values in undercharge.items():
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(values)), moving_avg, label=ev)
    plt.title(f'Undercharge [a={alpha}, B={beta}, y={gamma}]')
    plt.xlabel('Index')
    plt.ylabel('kW')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Undercharge_mean_{alpha}_{beta}_{gamma}.png'))
    plt.close()

    # Plot data from charge_att
    plt.figure(figsize=(6, 4))
    for ev, values in charging_rate_.items():
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(values)), moving_avg, label=ev)
    plt.title(f'Charge Rate [a={alpha}, B={beta}, y={gamma}]')
    plt.xlabel('Index')
    plt.ylabel('kW')
    plt.savefig(os.path.join(folder_name, f'charging_rate_{alpha}_{beta}_{gamma}.png'))
    plt.close()

    # Plot data from gap_time
    plt.figure(figsize=(6, 4))
    for ev, values in gap_time.items():
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(values)), moving_avg, label=ev)
    plt.title(f'Gap Time [a={alpha}, B={beta}, y={gamma}]')
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'gap_time_{alpha}_{beta}_{gamma}.png'))
    plt.close()

    # Plot data from complete_task
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(complete_charge)), complete_charge)
    plt.xticks(range(len(values)))
    plt.title(f'EVs Completion [a={alpha}, B={beta}, y={gamma}]')
    plt.xlabel('Index')
    plt.ylabel('EVs')
    plt.savefig(os.path.join(folder_name, f'complete_task_{alpha}_{beta}_{gamma}.png'))
    plt.close()