import numpy as np

class HER:
    def __init__(self, max_mem,input_shape, n_actions, goal_shape, batch_size,
                 reward_fn, strategy = 'final', k = 4):
        
        self.max_mem = max_mem
        self.strategy = strategy
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.reward_fn = reward_fn
        self.k = k
        
        self.states = np.zeros((max_mem, input_shape), dtype = np.float64)
        self.states_ = np.zeros((max_mem, input_shape), dtype = np.float64)
        self.actions = np.zeros((max_mem, n_actions), dtype = np.float32)
        self.rewards = np.zeros(max_mem, dtype = np.float32)
        self.dones = np.zeros(max_mem, dtype = bool)
        
        self.desired_goals = np.zeros((max_mem, goal_shape), dtype = np.float64)
        self.achieved_goals = np.zeros((max_mem, goal_shape), dtype = np.float64)
        self.achieved_goals_ = np.zeros((max_mem, goal_shape), dtype = np.float64)
        
        
    def store_memory(self, state, action, reward, states_, done, d_goal, a_goal, a_goal_):
        index = self.mem_cntr % self.max_mem
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = states_
        self.dones[index] = done
        self.desired_goals[index] = d_goal
        self.achieved_goals[index] = a_goal
        self.achieved_goals_[index] = a_goal_
        self.mem_cntr += 1
        
    def store_episode(self, ep_memory):
        states, actions, rewards, states_, dones, dg, ag, ag_ = ep_memory
        hindsight_goals = []
        
        if self.strategy == 'final':
            #achieved goals of last state of episode 
            hindsight_goals = [[ag_[-1]]]*len(ag_) 
            
        elif self.strategy == 'future':
            for idx, _ in enumerate(ag_):
                t_step_goals = []
                for m in range(self.k):
                    if idx + m >= len(ag_) - 1:
                        break
                    goal_idx = np.random.randint(idx + 1, len(ag_))
                    t_step_goals.append(ag_[goal_idx])
                hindsight_goals.append(t_step_goals)
                
        for idx, s in enumerate(states):
            self.store_memory(s, actions[idx], rewards[idx], states_[idx], dones[idx],
                              dg[idx], ag[idx], ag_[idx])
            
            for goal in hindsight_goals[idx]: 
            #for each state we have a hindsight goal
                reward = self.reward_fn(ag_[idx], goal, {})
                self.store_memory(s, actions[idx], reward, states_[idx], dones[idx],
                                  dg[idx], ag[idx], ag_[idx])
                
    def sample_memory(self):
        last_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(last_mem, self.batch_size, replace=False)
            
        return self.states[batch], self.actions[batch], self.rewards[batch], self.states_[batch], \
            self.dones[batch], self.desired_goals[batch]
        
    def ready(self):
        return self.mem_cntr > self.batch_size
