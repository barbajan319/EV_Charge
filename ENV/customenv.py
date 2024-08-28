import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
np.set_printoptions(suppress=True)
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=350, shape=(1,), dtype=np.float32)
  
        # Example for using image as input (channel-first; channel-last also works):
        low = np.array((0,0,0,0,0,0,0), dtype=np.float32)
        high = np.array((350,1,1,1,1,1,20), dtype=np.float32)
        
        
        self.observation_space = spaces.Box(
            low = low,
            high = high, 
            dtype= np.float32
        )
                                              
        # self.observation_space = spaces.Box(low=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
        #                                     high=np.array([345.0,1.0,1.0,1.0,1.0,1.0,1.0,20]),
        #                                     shape=(8,), dtype=np.float32)
        # Initialize other instance variables here

    def step(self, action):
        overcharge_penalty = 0
        energy_consumed = self.charge_EV(action, self.resolution)   
        
        self.SOC +=  energy_consumed
        
        if self.SOC > self.target_SOC:
            self.done = True 
            overcharge_penalty = self.SOC - self.target_SOC
            
        self.SOC = min(self.SOC, self.target_SOC)
        charging_cost = self.calculate_charging_cost(energy_consumed, self.timestep, self.energy_prices)
        self.charging_cost += charging_cost
        self.timestep += 1
        
        if self.timestep == self.departure_time:
            self.done = True
        
        if self.done:
            if self.SOC < self.target_SOC:
                reward = -self.charging_cost - 100 
            else:  
                reward = -self.charging_cost - overcharge_penalty
        
        current_energy_cost = self.energy_prices[self.timestep]
        SOC = self.SOC
        energy_1_ahead = self.energy_prices[(self.timestep + 1)]
        energy_2_ahead = self.energy_prices[(self.timestep + 2)]
        energy_3_ahead = self.energy_prices[(self.timestep + 3)]
        energy_4_ahead = self.energy_prices[(self.timestep + 4)]
        
        time_left_depot =(self.departure_time - self.timestep)
        
        self.obs = np.array([SOC, current_energy_cost, energy_1_ahead, energy_2_ahead, energy_3_ahead, energy_4_ahead, time_left_depot], dtype=np.float32)
        info = {}
        return self.obs, self.reward, self.done, self.truncated, info

    def reset(self, seed=None):
        self.done = False
        self.resolution = 15 
        self.SOC = round(random.uniform(35, 210),3)
        self.energy_prices = self.set_price_array(1, self.resolution, random.choice([1, 2, 3, 4]))
        self.charging_cost = 0
        self.timestep = 0
        self.departure_time = random.randint(5, 18)
        self.target_SOC = 345
        self.reward = 0
        self.truncated = False
        
        current_energy_cost = self.energy_prices[self.timestep]
        
        SOC = self.SOC
        energy_1_ahead = self.energy_prices[(self.timestep + 1)]
        energy_2_ahead = self.energy_prices[(self.timestep + 2)]
        energy_3_ahead = self.energy_prices[(self.timestep + 3)]
        energy_4_ahead = self.energy_prices[(self.timestep + 4)]
        
        time_left_depot = (self.departure_time - self.timestep)
        
        self.obs = np.array([SOC, current_energy_cost, energy_1_ahead, energy_2_ahead, energy_3_ahead, energy_4_ahead, time_left_depot], dtype=np.float32)
        info = {}
        
        return self.obs, info

    @staticmethod
    def charge_EV(action, resolution):
        action = action[0]
        charging_rate = action  
        return charging_rate * (resolution / 60)
    
    @staticmethod
    def calculate_charging_cost(energy_consumption, timestep, energy_prices):
        energy_cost = energy_prices[timestep]
        charging_cost = energy_cost * energy_consumption
        return charging_cost
    @staticmethod
    def set_price_array(days_of_experiment, resolution, flag):
        intervals_per_hour = 60 // resolution
        intervals_per_day =  intervals_per_hour * 24
        intervals_total = days_of_experiment  * intervals_per_day
        price_array = np.zeros((intervals_total))
        price_flag = flag

        price_day = []
        
        if price_flag==1:
            Price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        elif price_flag==2:
            Price_day=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08 ,0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06 ,0.06 ,0.06, 0.05, 0.05, 0.05])
        elif price_flag==3:
            Price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080, 0.080, 0.1, 0.1, 0.076, 0.076,
                        0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
        elif price_flag==4:
            Price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1,
                        0.1, 0.1])
        
        repeated_hourly_prices = np.repeat(Price_day, intervals_per_hour)
        # Calculate scaling factor for each interval
        scaling_factor = 1 / intervals_per_hour
        # Interpolate prices for smaller intervals
        Energy_Prices = repeated_hourly_prices * scaling_factor
        return Energy_Prices
    
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import random
# np.set_printoptions(suppress=True)
# class CustomEnv(gym.Env):
#     """Custom Environment that follows gym interface"""
#     def __init__(self):
#         super(CustomEnv, self).__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Tuple((spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32), spaces.Discrete(5)))

  
#         # Example for using image as input (channel-first; channel-last also works):
#         low = np.array((0,0,0,0,0,0,0), dtype=np.float32)
#         high = np.array((350,1,1,1,1,1,20), dtype=np.float32)
        
        
#         self.observation_space = spaces.Box(
#             low = low,
#             high = high, 
#             dtype= np.float32
#         )
                                              
#         # self.observation_space = spaces.Box(low=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
#         #                                     high=np.array([345.0,1.0,1.0,1.0,1.0,1.0,1.0,20]),
#         #                                     shape=(8,), dtype=np.float32)
#         # Initialize other instance variables here

#     def step(self, action):
#         charge_rate, charging_decision = action
        
#         charge_rate = charge_rate[0]
        
#         overcharge_penalty = 0
#         energy_consumed = self.charge_EV(charge_rate, self.resolution)   
        
#         self.SOC +=  energy_consumed
        
#         if self.SOC > self.target_SOC:
#             self.done = True 
#             overcharge_penalty = self.SOC - self.target_SOC
            
#         self.SOC = min(self.SOC, self.target_SOC)
#         charging_cost = self.calculate_charging_cost(energy_consumed, self.timestep, self.energy_prices)
#         self.charging_cost += charging_cost
#         self.timestep += 1
        
#         if self.timestep == self.departure_time:
#             self.done = True
        
#         if self.done:
#             if self.SOC < self.target_SOC:
#                 reward = -self.charging_cost - 100 
#             else:  
#                 reward = -self.charging_cost - overcharge_penalty
        
#         current_energy_cost = self.energy_prices[self.timestep]
#         SOC = self.SOC
#         energy_1_ahead = self.energy_prices[(self.timestep + 1)]
#         energy_2_ahead = self.energy_prices[(self.timestep + 2)]
#         energy_3_ahead = self.energy_prices[(self.timestep + 3)]
#         energy_4_ahead = self.energy_prices[(self.timestep + 4)]
        
#         time_left_depot =(self.departure_time - self.timestep)
        
#         self.obs = np.array([SOC, current_energy_cost, energy_1_ahead, energy_2_ahead, energy_3_ahead, energy_4_ahead, time_left_depot], dtype=np.float32)
#         info = {}
#         return self.obs, self.reward, self.done, self.truncated, info

#     def reset(self, seed=None):
#         self.done = False
#         self.resolution = 15 
#         self.SOC = round(random.uniform(35, 210),3)
#         self.energy_prices = self.set_price_array(1, self.resolution, random.choice([1, 2, 3, 4]))
#         self.charging_cost = 0
#         self.timestep = 0
#         self.departure_time = random.randint(5, 18)
#         self.target_SOC = 345
#         self.reward = 0
#         self.truncated = False
        
#         current_energy_cost = self.energy_prices[self.timestep]
        
#         SOC = self.SOC
#         energy_1_ahead = self.energy_prices[(self.timestep + 1)]
#         energy_2_ahead = self.energy_prices[(self.timestep + 2)]
#         energy_3_ahead = self.energy_prices[(self.timestep + 3)]
#         energy_4_ahead = self.energy_prices[(self.timestep + 4)]
        
#         time_left_depot = (self.departure_time - self.timestep)
        
#         self.obs = np.array([SOC, current_energy_cost, energy_1_ahead, energy_2_ahead, energy_3_ahead, energy_4_ahead, time_left_depot], dtype=np.float32)
#         info = {}
        
#         return self.obs, info

#     @staticmethod
#     def charge_EV(action, resolution):
#         action = action[0]
#         charging_rate = action  
#         return charging_rate * (resolution / 60)
    
#     @staticmethod
#     def calculate_charging_cost(energy_consumption, timestep, energy_prices):
#         energy_cost = energy_prices[timestep]
#         charging_cost = energy_cost * energy_consumption
#         return charging_cost
#     @staticmethod
#     def set_price_array(days_of_experiment, resolution, flag):
#         intervals_per_hour = 60 // resolution
#         intervals_per_day =  intervals_per_hour * 24
#         intervals_total = days_of_experiment  * intervals_per_day
#         price_array = np.zeros((intervals_total))
#         price_flag = flag

#         price_day = []
        
#         if price_flag==1:
#             Price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
#                         0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
#         elif price_flag==2:
#             Price_day=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08 ,0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06 ,0.06 ,0.06, 0.05, 0.05, 0.05])
#         elif price_flag==3:
#             Price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080, 0.080, 0.1, 0.1, 0.076, 0.076,
#                         0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
#         elif price_flag==4:
#             Price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1,
#                         0.1, 0.1])
        
#         repeated_hourly_prices = np.repeat(Price_day, intervals_per_hour)
#         # Calculate scaling factor for each interval
#         scaling_factor = 1 / intervals_per_hour
#         # Interpolate prices for smaller intervals
#         Energy_Prices = repeated_hourly_prices * scaling_factor
#         return Energy_Prices
