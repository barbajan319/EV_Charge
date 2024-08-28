import numpy as np
from EVenv import MultiAgentEVCharge
import os
import random
import math
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions_1, n_actions_2, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_counter = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions_1 + n_actions_2
        self.actor_dims = actor_dims
        # State Memory
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        # For critic value for the terminal state is always 0, when an agent reaches a terminal state
        # the episode terminates and no future rewards follow
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
        # Iterate over all agents getting observations and storing them to the appropriate agent's actor
        for agent_idx in range(self.n_agents):
            state_vec = list(raw_obs["EV_%s" % agent_idx].values())
            state_vec_ = list(new_raw_obs["EV_%s" % agent_idx].values())
            vector = np.concatenate([arr.flatten() for arr in state_vec])
            vector_ = np.concatenate([arr.flatten() for arr in state_vec_])

            action_1, action_2 = action[agent_idx]
            #Maybe change to array
            actions = np.concatenate([action_1, action_2])

            if len(vector) != self.actor_dims[agent_idx]:
                zero_pad = self.actor_dims[0] - len(vector)
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

    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
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

class CriticNetwork(Model):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions_1, n_actions_2, name, chkpt_dir):
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(64, activation='relu')
        self.q = Dense(1)

        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=beta)

    def forward(self, state, action):
        X = tf.concat([state, action], axis=1)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        q = self.q(X)
        return q
    
    def save_checkpoint(self):
        # Save model weights
        self.save_weights(self.chkpt_file)

    def load_checkpoint(self):
        # Load model weights
        self.load_weights(self.chkpt_file)

class ActorNetwork(Model):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions_1, n_actions_2):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(64, activation='relu')

        self.continuous_output = Dense(n_actions_1, activation='sigmoid')
        self.discrete_output = Dense(n_actions_2, activation='softmax')

        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=alpha)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        charge_rate = self.continuous_output(x)
        charge_decision = self.discrete_output(x)

        return charge_rate, charge_decision
    
    def save_checkpoint(self):
        self.save_weights(self.chkpt_file)
        
    def load_checkpoint(self):
        self.load_weights(self.chkpt_file)
        
        
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
    
    self.update_network_parameters(tau = self.tau)

    def update_network_parameters(self, tau=0.01):
        
        self.target_actor.set_weights(tau * np.array(self.actor.get_weights()) +
                                       (1 - tau) * np.array(self.target_actor.get_weights()))

        self.target_critic.set_weights(tau * np.array(self.critic.get_weights()) +
                                        (1 - tau) * np.array(self.target_critic.get_weights()))

    def choose_action(self, observation):
        soc = observation['SOC']/450
        target_soc = observation['Target_SOC']/450
        time_left_depot = observation['Time_left_Depot']/96
        current_energy_price = observation['Current_Energy_Price']
        future_energy_window = observation['Future_Energy_Window']
        load_on_transformer = observation['Load_on_transformer']/1350
        available_stations = observation['Available_Stations']
        
        state = tf.concat([soc, target_soc, time_left_depot, current_energy_price,
                   future_energy_window, available_stations, load_on_transformer], axis=-1)
        
        charge_rate, charge_decision = self.actor.forward(state)
        epsilon = 0.1
        noise_level = 0.5
        
        if random.random() < epsilon:
            noisy_logits = tf.math.log(charge_decision) + tf.random.normal(shape=tf.shape(charge_decision)) * noise_level
            charge_decision = tf.nn.softmax(noisy_logits, axis=-1)
            noise = tf.random.uniform(shape=tf.shape(charge_rate))
            charge_rate = charge_rate + noise
            charge_rate =  tf.clip_by_value(input_tensor, clip_value_min=0, clip_value_max=1)
        
        return charge_rate, charge_decision
    
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

  def save_checkpoint(self):
    print('....saving checkpoint.....')
    for agent in self.agents:
      agent.save_models()
      
  def load_checkpoint(self):
    print('....loading checkpoint.....')
    for agent in self.agents:
      agent.load_models()
      
      
    def choose_action(self, obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            actions.append(agent.choose_action(obs["EV_%s" % agent_idx]))
        return actions

    def learn(self):
        if not self.memory.ready():
            return

        actor_states, states, actions, rewards, actor_new_states, states_, terminations = memory.sample_buffer()

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_np = np.array(actions, dtype=np.float32)
        actions = tf.convert_to_tensor(actions_np, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = T.tensor(states_, dtype = T.float32).to(device)
        terminations = T.tensor(terminations).to(device)
        
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        
        for agent_idx, agent in enumerate(self.agents):
            #estimate action values for the next state according to actor network
            new_states = tf.convert_to_tensor(actor_new_states[agent_idx])
            new_charge_rate, new_charge_decision = agent.target_actor.forward(new_states)
            all_agents_new_actions.append((new_charge_rate))
            all_agents_new_actions.append((new_charge_decision))
            
            mu_states = tf.convert_to_tensor(actor_states[agent_idx], dtype = tf.float32)
            charge_rate, charge_decision = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append((charge_rate))
            all_agents_new_mu_actions.append((charge_decision))
            
            old_agents_actions.append(actions[agent_idx])
        
        new_actions = tf.concat([acts for acts in all_agents_new_actions], dim = 1)
        mu = tf.concat([acts for acts in all_agents_new_mu_actions], dim = 1)
        old_actions =  tf.concat([acts for acts in old_agents_actions], dim = 1)
        
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[terminations[:,0]] = 0.0
            
            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            mean_rewards = tf.reduce_mean(rewards[:, agent_idx])
            std_rewards = tf.math.reduce_std(rewards[:, agent_idx])
            normalized_rewards = (rewards[:, agent_idx] - mean_rewards) / (std_rewards + 1e-8)
            
            target = normalized_rewards + agent.gamma*critic_value_
            
            with tf.GradientTape() as tape:
                critic_loss = tf.keras.losses.MSE(target, critic_value)
            
            self.writer.add_scalar(f"EV_{agent_idx}/Loss/Critic", critic_loss, episode)
            critic_gradients = tape.gradient(critic_loss, agent.critic.trainable_variables)
            agent.critic.optimizer.apply_gradients(zip(critic_gradients, agent.critic.trainable_variables))

            actor_loss = agent.critic.forward(states, mu).flatten()
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(actor_loss)
                
            self.writer.add_scalar(f"EV_{agent_idx}/Loss/Actor", critic_loss, episode)
            actor_gradients = tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.critic.optimizer.apply_gradients(zip(actor_gradients, agent.actor.trainable_variables))

            agent.update_network_parameters()
            
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