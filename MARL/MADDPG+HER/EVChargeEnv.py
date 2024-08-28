import numpy as np
import gym 

class EVCharge:
    def __init__(self, max_charge = 450, max_rate = 350, buffer = 10, max_steps = 96):
        self.max_steps = max_steps
        self.max_charge = max_charge
        self.target = 0.9 * max_charge
        self.max_rate = max_rate
        self.buffer = buffer
        self.n_actions = 1
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space
        
        self.observation = self.reset()
        
        self.observation_space = {
            'observation': np.empty((2,)),
            'achieved_goal': np.empty((1,)),
            'desired_goal': np.empty((1,)),  
        }
        
    def reset(self):
        self._battery_level = np.random.randint(0, self.max_charge * 0.6)
        self._desired_goal = np.random.randint(self._battery_level, self.target + 1)
        self._achieved_goal = self._battery_level
        self._departure_times = np.random.randint(10, self.max_steps)
        
        obs = np.array([self._battery_level, self._departure_times, self._achieved_goal, self._desired_goal])
        
        self._step = 0
        return obs
    
    def compute_reward(self, desired_goal, achieved_goal, info):
        reward = 0.0 if (desired_goal <= achieved_goal <= desired_goal + self.buffer) else -1.0 
        return reward
    
    def step(self, action):
        if self._step >= self._departure_times:
            action = 0
        assert -1 <= action <= 1, "Invalid Action"
        rate = action * 350
        
        if rate > 0:
            self._battery_level = self.chargeEV(rate, self._battery_level, self.max_steps)
            
        info = {}
        self._achieved_goal = self._battery_level
        reward = self.compute_reward(self._desired_goal, self._achieved_goal, {})
        
        self._step += 1
        if reward == 0.0 or self._step >= self.max_steps:
            done = True 
        else:
            done = False
            
        time_left_depot = self._departure_times - self._step
        
        info['is_success'] = 1.0 if reward == 0.0 else 0.0 
        obs = np.array([self._battery_level, time_left_depot,  self._achieved_goal, self._desired_goal])
        
        return obs, reward, done, info
    
    def action_space_sample(self):
        return np.random.uniform(-1, 1)
    
    def render(self):
        print(f'Battery Level: {self._battery_level}, Target: {self._desired_goal}, Step: {self._step}')
        
    def chargeEV(self, rate, battery, timesteps):
        resolution = 60/(timesteps/24)
        charge = rate * (resolution/60)
        new_battery_level = battery + charge 
        new_battery_level = np.clip(new_battery_level, 0. , self.max_charge )
        return new_battery_level