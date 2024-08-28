from pettingzoo import  ParallelEnv
from pettingzoo.utils import wrappers
import gymnasium 
from gymnasium.spaces import Box, Tuple, Discrete, Dict, MultiBinary, MultiDiscrete
import numpy as np
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import random
import copy
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils import BaseParallelWrapper
import time
import torch as T


def get_target_SOC(upcoming_millage, battery_capacity):
    nominal_efficiency = 2.2 #Estimated overall operating efficiency of the vehicle: 2.2 KWh / mile
    # Peak Power = Maximum power generated by the vehicle's motor: 250 KW
    
    energy_needed = upcoming_millage * nominal_efficiency
    buffer = battery_capacity * .10

    target_soc = energy_needed + buffer
    target_soc = min(target_soc, battery_capacity)
    return target_soc

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
    return np.array(Energy_Prices)

def generate_departure_time(timestep, max_timestep):
    return random.randint(timestep, (max_timestep - 1))

def remove_agent(agents, depart_agents, EV):
    agents.remove(EV)
    depart_agents.append(EV)
    
def agent_reentry(depart_agents, EV, SOC, next_millage, target_SOC, agents, terminations):
    reentry_probability = 0.2
    for EV in depart_agents:
        if random.random( ) < reentry_probability:
            depart_agents.remove(EV)
            agents.append(EV)
            terminations[EV] = False
            SOC[EV] = random.uniform(45, 180)
            next_millage[EV] =  random.randint(60, 280)
            target_SOC[EV] = get_target_SOC(next_millage[EV], battery_capacity=450)
            

def charge_EV(rate, resolution):
        return rate * (resolution / 60) 
    
def calculate_charging_cost(energy_consumption, timestep, energy_prices):
        energy_cost = energy_prices[timestep]
        charging_cost = energy_cost * energy_consumption
        return charging_cost
    
def find_duplicate_ev(ev_station_assignment):
    # Create a reverse dictionary where keys are stations and values are lists of EVs
    station_ev_map = {}
    for ev, station in ev_station_assignment.items():
        if station is not None:
            station_ev_map.setdefault(station, []).append(ev)

    # Initialize a list to store EVs assigned to duplicate stations
    duplicate_evs = []

    # Iterate over the station_ev_map to find duplicate stations
    for station, ev_list in station_ev_map.items():
        if len(ev_list) > 1:
            duplicate_evs.extend(ev_list)

    return duplicate_evs

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    return env

def waiting_reward(target_soc, soc, time_left):
    soc = soc + 0.01
    reward = (1/(target_soc - soc)) * time_left
    return reward

class MultiAgentEVCharge_V4(ParallelEnv):
    metadata = {
        "name": "Depot_env_v0"
    }
    
    def __init__(self, num_agents, render_mode = None):
        self.possible_agents = ["EV_" + str(r) for r in range(num_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # self.num_stations = random.randint(1, len(self.possible_agents))
        self.resolution = 15
        self.num_intervals = int((60/self.resolution)*24)
        self.battery_capacity = 450
        self.alpha = 0.1
        self.beta = 0.001
        self.gamma = 1
        self.delta = 0
        self.transformer_capacity = 1350 #1500kVa


    def reset(self, seed=None, options=None):
        #reset the environment to start again
        self.contributions  = {'EV_0': {'undercharge': [0,0.0], 'overcharge_bat': [0,[]], 'overcharge_tar': [0,0], 'gap': [0,0]}, 
                               'EV_1': {'undercharge': [0,0.0], 'overcharge_bat': [0,[]], 'overcharge_tar': [0,0], 'gap': [0,0]}, 
                               'EV_2': {'undercharge': [0,0.0], 'overcharge_bat': [0,[]], 'overcharge_tar': [0,0], 'gap': [0,0]}, 
                               'EV_3': {'undercharge': [0,0.0], 'overcharge_bat': [0,[]], 'overcharge_tar': [0,0], 'gap': [0,0]}}        
        flag = random.choice([1, 2, 3, 4])
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.cummulative_rewards = {agent: 0 for agent in self.agents}
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = False
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.indv_task = {agent: False for agent in self.agents}
        self.cummulative_charging_costs = 0
        self.completed_task_reward = 0
        self.task_completion = False
        self.group_reward_received = False 
        
        self.timestep  = 0
        self.transformer_load = 0 
        self.num_stations = 4 #random.randint(1, len(self.agents))
        self.charging_station_availability = {station: True for station in range(self.num_stations)} 
    
        #Initialize EVs battery randomly between 10% and 60%
        self.SOC = {agent: random.uniform(45, 270) for agent in self.agents}
        #Determine each EVs next trip millage (Based on range of miles travelled by sandhaul)
        self.next_millage = {agent: random.randint(60, 205) for agent in self.agents} 
        #With the millage set the target SOC for next trip 
        self.target_SOC = {agent: get_target_SOC(self.next_millage[agent], self.battery_capacity) for agent in self.agents}
        #Randomize when EVs will depart for their first trip and time left at the depot
        self.departure_times = {agent: generate_departure_time(self.timestep + 1, self.num_intervals) for agent in self.agents}
        self.energy_prices = set_price_array(1, self.resolution, flag)
        self.assigned_station = {agent: None for agent in self.agents}
        self.charging_attempt = {agent: 0 for agent in self.agents}
        self.complete_charge = 0
        self.gap_time = {agent: 0 for agent in self.agents}
        
        self.action_spaces = {
            agent: Tuple((
                Box(low=0, high=350, shape=(1,), dtype=np.float32),
                Discrete(2)
            ))     
           for agent in self.agents 
        }
        
        self.observation_spaces = {
            agent: Dict(
                {
            "SOC": Box(low=0, high=350, shape=(1,), dtype=np.float32),
            "Target_SOC": Box(low=0, high=350, shape=(1,), dtype=np.float32),
            "Time_left_Depot": Discrete(self.num_intervals),
            # "Current_Energy_Price": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # "Future_Energy_Window": Box(low=0, high=1, shape=(len(self.energy_prices),), dtype=np.float32),
            # "Load_on_transformer": Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "Available_Stations": MultiBinary(self.num_stations),
            "SOC_all_agents": MultiDiscrete([1,1,1]),
            "Target_all_agents":  MultiDiscrete([1,1,1]),
            "Time_Left_All_Agents":  MultiDiscrete([1,1,1])
        })
            for agent in self.possible_agents
            }
        

        self.departed_agents = []


        charging_stations_available = [1 if value else 0 for value in self.charging_station_availability.values()]
     
        for agent in self.agents:
            time_left_depot_all = []
            soc_all = []
            target_soc_all = []
            for other_agent in self.possible_agents:
                if other_agent != agent:
                    time_left_depot_all.append(self.departure_times[other_agent])
                    soc_all.append(self.SOC[other_agent])
                    target_soc_all.append(self.target_SOC[other_agent])
            self.observations[agent] = {
            "SOC": np.array([self.SOC[agent]], dtype=np.float32),
            "Target_SOC": np.array([self.target_SOC[agent]], dtype=np.float32),
            "Time_left_Depot": np.array([self.departure_times[agent]], dtype=np.float32),
            "Available_Stations": np.array(charging_stations_available),
            "SOC_all_agents": np.array(soc_all),
            "Target_all_agents": np.array(target_soc_all),
            "Time_Left_All_Agents": np.array(time_left_depot_all)
            }
            
        return self.observations, self.infos
    
    def step(self, actions):
        charging_stations_available = [1 if value else 0 for value in self.charging_station_availability.values()]
        infos = {agent: {} for agent in self.agents}
        agent_actions = {agent: [0,0] for agent in self.possible_agents}
        sufficient_battery_reward = 1000
        overcharge_penalty = {agent: 0 for agent in self.possible_agents}
        undercharge_penalty = {agent: 0 for agent in self.possible_agents}
        gap_penalty = {agent: 0 for agent in self.possible_agents}
        task_rewards = {agent: 0 for agent in self.possible_agents}
        target_met_reward = {agent: 0 for agent in self.possible_agents}

        
        
        #Depart all agents that need to leave  
        for agent in list(self.agents)[:]: 
            #Determine if agent needs to depart
            if self.timestep == self.departure_times[agent]:
                #Check if agent is asigned to a station, to make sure station becomes available after departure
                if self.assigned_station[agent] != None: 
                    station = self.assigned_station[agent] #Get station at which agent is currently assigned
                    self.charging_station_availability[station] = True  #Set the charging station as ava
                    self.assigned_station[agent] = None
                self.agents.remove(agent)
                self.departed_agents.append(agent)
        #Iterate through agents and remove from station those who will not charge to make station available
        for agent in list(self.agents)[:]: 
            #for each agent extract their actions:
            agent_idx = int(agent.split('_')[1])
            charge_rate, charge_decision = actions[agent_idx]
            charge_decision = T.argmax(charge_decision).item()
            charge_rate = charge_rate[0] * 350
            agent_actions[agent]  = [charge_rate, charge_decision]
            
            #check if target SOC was met 
            if self.timestep == self.departure_times[agent] - 1:
                #If Target_SOC is not met assign an undercharged penalty
                if self.SOC[agent] < self.target_SOC[agent]:
                    undercharge_penalty[agent] = ((self.target_SOC[agent] - self.SOC[agent])**2) * self.alpha        
                    self.contributions[agent]['undercharge'][0] += 1
                    self.contributions[agent]['undercharge'][1] += (self.target_SOC[agent] - self.SOC[agent])
                    # self.rewards[agent] -= undercharge_penalty
                    
                else:
                    #give reward for completing task
                    # self.rewards[agent] += sufficient_battery_reward
                    #Keep track of agents who completed a task
                    self.complete_charge += 1
                    self.indv_task[agent] = True
                    task_rewards[agent] += sufficient_battery_reward
            
            if self.assigned_station[agent] is  None and charge_decision == 1:
                self.charging_attempt[agent] += 1
            
            
            if self.assigned_station[agent] != None: #Meaning agent is assigned to a station 
                if charge_decision == 0:
                #Remove agent from charging station and make available
                    station = self.assigned_station[agent]
                    self.assigned_station[agent] = None
                    self.charging_station_availability[station] = True
                    #give a reward  for leaving a charging station
                    # time_left = self.departure_times[agent] - self.timestep
                    # leave_station_reward = waiting_reward(self.target_SOC[agent], self.SOC[agent], time_left)
                    # self.rewards[agent] += leave_station_reward


                #give a penalty if the agent decides to charge but has reached its target SOC
                elif  charge_decision != 0 and self.SOC[agent] >=  self.target_SOC[agent]:
                    overcharge_penalty[agent] += 1
                    self.contributions[agent]['overcharge_tar'][0] += 1
                    self.contributions[agent]['overcharge_tar'][1] += 1
            

                    # self.rewards[agent] -= 2
        #Look through the agents that want to book a station and make sure there are enough stations
        #If available stations <= agents wanting to charge, all charge, else none charge
        charge_request = []
        for agent in self.assigned_station:
            #check if agent is not assigned to station and wants to charge
            if self.assigned_station[agent] is None and agent_actions[agent][1] == 1:
                charge_request.append(agent)
        #check if number of stations can accomodate number of requests
        if len(charge_request) <= sum(value == True for value in self.charging_station_availability.values()):
            #If true assign agents to station
            for request in charge_request:
                for station, available in self.charging_station_availability.items():
                    if available:
                        self.charging_station_availability[station] = False
                        self.assigned_station[request] = station
                        break
    

        #Else penalize 
        # else:
        #     for request in charge_request:
        #         self.rewards[request] =- 1
        #         print("request penalty ")
                    
                    
        #Update charge of charging agents
        for agent in list(self.agents)[:]: 
            if self.assigned_station[agent] != None:
                charge_rate_agent = agent_actions[agent][0]
                energy_consumed = charge_EV(charge_rate_agent, self.resolution)
                self.SOC[agent] += energy_consumed
                #Penalize if charging at a rate that would lead to over charging
                if self.SOC[agent] > self.battery_capacity:
                    overcharge_penalty[agent] += ((self.SOC[agent] - self.battery_capacity)*charge_rate_agent) * self.beta

                    # self.rewards[agent] -= overcharge_penalty
                    self.contributions[agent]['overcharge_bat'][0] += 1
                    self.contributions[agent]['overcharge_bat'][1].append((self.SOC[agent] - self.battery_capacity))
                    self.SOC[agent] =  self.battery_capacity

            #reward for closing the gap between  SOC and target SOC
            if self.SOC[agent] < self.target_SOC[agent]:
                gap_penalty[agent] += ((1 - (self.SOC[agent]/self.target_SOC[agent])) / (1 - (self.timestep/self.departure_times[agent]))) * 20
                self.contributions[agent]['gap'][0] += 1
                self.contributions[agent]['gap'][1] +=  gap_penalty[agent]
                
                
        #     else:
        #         #If not assigned to a station, then the agent might be searching for a charger
        #         if charge_decision == 0:
        #             time_left = self.departure_times[agent] - self.timestep
        #             leave_station_reward = waiting_reward(self.target_SOC[agent], self.SOC[agent], time_left)
        #             self.rewards[agent] += leave_station_reward
                    

        for station in self.charging_station_availability:
            if station not in self.assigned_station.values():
                self.charging_station_availability[station] = True
            else:
                self.charging_station_availability[station] = False
                
        
                  
        self.timestep += 1
        remaining_timesteps = len(self.energy_prices) - self.timestep
        future_energy_window_array = np.full(len(self.energy_prices), fill_value=-1, dtype=np.float32)
        future_energy_window_array[self.timestep:] = self.energy_prices[:remaining_timesteps]
        charging_stations_available = [1 if value else 0 for value in self.charging_station_availability.values()]
        #set the new observations for the next timestep
        for agent in self.agents:
            time_left_depot_all = []
            soc_all = []
            target_soc_all = []
            for other_agent in self.possible_agents:
                if other_agent != agent:
                    time_left_depot = (self.departure_times[other_agent] - self.timestep)/self.num_intervals
                    time_left_depot_all.append(time_left_depot)
                    soc_all.append(self.SOC[other_agent]/self.battery_capacity)
                    target_soc_all.append(self.target_SOC[other_agent]/self.battery_capacity)
            time_left_depot = (self.departure_times[agent] - self.timestep)/self.num_intervals
            self.observations[agent] = {
                "SOC": np.array([self.SOC[agent]/self.battery_capacity], dtype=np.float32),
                "Target_SOC": np.array([self.target_SOC[agent]/self.battery_capacity], dtype=np.float32),
                "Time_left_Depot": np.array([time_left_depot], dtype=np.float32),
                "Available_Stations": np.array(charging_stations_available),
                "SOC_all_agents": np.array(soc_all),
                "Target_all_agents": np.array(target_soc_all),
                "Time_Left_All_Agents": np.array(time_left_depot_all)
            }
            
        if self.complete_charge == len(self.possible_agents) and not self.group_reward_received:
            #Global reward for all  agents combined
            self.task_completion = True
            global_reward = 5000
            
            if self.group_reward_received:
                self.rewards = self.cummulative_rewards
                self.terminations = {agent: True for agent in self.possible_agents}
                self.agents.clear()
                self.truncations = True
                break
            
            for agent in self.possible_agents:
                task_rewards[agent] += global_reward
                rewards = task_rewards[agent] + target_met_reward[agent] - undercharge_penalty[agent] - overcharge_penalty[agent] - gap_penalty[agent]
                self.cummulative_rewards[agent] += rewards
                self.rewards[agent] = self.cummulative_rewards[agent]
                
            self.group_reward_received = True
        
        elif self.group_reward_received:
            self.rewards = self.cummulative_rewards
            self.terminations = {agent: True for agent in self.possible_agents}
            self.agents.clear()
            self.truncations = True
        
        else:   
            for agent in self.possible_agents:
                self.rewards[agent] = task_rewards[agent] + target_met_reward[agent] - undercharge_penalty[agent] - overcharge_penalty[agent] - gap_penalty[agent]
                self.cummulative_rewards[agent] += self.rewards[agent]
            
        if self.timestep == self.num_intervals - 1:
            for agent in self.possible_agents:
                self.rewards[agent] = self.cummulative_rewards[agent]  

                
        if self.timestep == self.num_intervals:
            self.terminations = {agent: True for agent in self.possible_agents}
            self.agents.clear()
            self.truncations = True
            self.rewards = self.cummulative_rewards
            
                  
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos, self.task_completion
    
    def render(self):
        import tkinter as tk

        # Define the information for each EV
        ev_info = []
        
        for agent in self.agents:
            if self.assigned_station[agent] is not None:
                status = "Charging"
                assigned_station = self.assigned_station[agent] + 1
            else:
                status = "Idle"
                assigned_station  = None
                
            ev_dict = {"name": agent, "charge": round((self.SOC[agent] / self.battery_capacity) * 100, 1), "status": status, "station": assigned_station, "Target SOC": round((self.target_SOC[agent] / self.battery_capacity) * 100, 1), "Departure": self.departure_times[agent] }
            ev_info.append(ev_dict)
            
        # Define the available charging stations
        charging_stations = np.arange(1, self.num_stations + 1)

        # Function to get color code based on charge level
        def get_color(charge):
            if charge < 30:
                return "red"
            elif charge >= 30 and charge < 75:
                return "orange"
            else:
                return "green"

        # Create the main window
        root = tk.Tk()
        root.title("EV Information")

        # Define a function to create the grid layout
        def create_grid():
            # Loop over each EV and create a label with its information and meter visualization
            for i, ev in enumerate(ev_info):
                # Determine the row and column for the label
                row = i // 2
                col = i % 2
                
                # Create a frame to contain EV information and meter
                frame = tk.Frame(root, borderwidth=2, relief="solid")
                frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Create a label with EV information
                if ev["status"] == "Charging":
                    label_text = f"{ev['name']}:\nCharge: {ev['charge']}%\nStatus: {ev['status']}\nStation: {ev['station']}\nTarget: {ev['Target SOC']}% \nDeparture: {ev['Departure']}"
                else:
                    label_text = f"{ev['name']}:\nCharge: {ev['charge']}%\nStatus: {ev['status']}\nTarget: {ev['Target SOC']} \nDeparture: {ev['Departure']}"
                label = tk.Label(frame, text=label_text, padx=10, pady=5, wraplength=150, anchor="center")
                label.grid(row=0, column=0, sticky="nsew")
                
                # Create a canvas for the meter visualization
                canvas = tk.Canvas(frame, width=100, height=20)
                canvas.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
                
                # Draw the meter background
                canvas.create_rectangle(0, 0, 100, 20, fill="lightgray", outline="")
                
                # Draw the meter fill based on charge level
                color = get_color(ev["charge"])
                canvas.create_rectangle(0, 0, ev["charge"], 20, fill=color, outline="")

        # Define a function to create the column for available charging stations
        def create_station_column():
            # Check if each charging station is occupied by an EV
            station_text = "Available Charging Stations:\n"
            for station in charging_stations:
                occupied = any(ev["station"] == station for ev in ev_info)
                availability = "Occupied" if occupied else "Available"
                station_text += f"Station {station}: {availability}\n"
            
            # Create a label to display the charging station information
            station_label = tk.Label(root, text=station_text, borderwidth=2, relief="solid", padx=10, pady=5, wraplength=150, anchor="center")
            station_label.grid(row=0, column=2, rowspan=len(ev_info) // 2 + 1, padx=5, pady=5, sticky="nsew")
        # Function to update the time step display
    # Function to update the time step display
        def update_time_step():
          # Convert timestep to hour of the day
            time_label.config(text=f"Current Timestep: {self.timestep}")
            
        
        root.after(500, update_time_step)  # Update every second
        # Call the functions to create the grid layout and the station column
        create_grid()
        create_station_column()
        
        time_label = tk.Label(root, text="", padx=5, pady=2, wraplength=100, anchor="center", font=("Arial", 10))
        time_label.grid(row=0, column=0, columnspan=2, sticky="ne")
        update_time_step()

        # Configure row and column weights to make them resize with the window
        for i in range(2):
            root.grid_columnconfigure(i, weight=1)
        for i in range((len(ev_info) + 1) // 2):
            root.grid_rowconfigure(i, weight=1)

        # Start the Tkinter event loop
        
        w = 800 # width for the Tk root
        h = 600 # height for the Tk root

        # get screen width and height
        ws = root.winfo_screenwidth() # width of the screen
        hs = root.winfo_screenheight() # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        # set the dimensions of the screen 
        # and where it is placed
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

 
        root.after(8000, lambda: root.destroy()) # Destroy the widget after 30 seconds

        root.mainloop()

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        #return Tuple((Box(low=0, high=350, shape=(1,), dtype=np.float32),Discrete(num_stations + 1)))
        return self.action_spaces[agent]
        