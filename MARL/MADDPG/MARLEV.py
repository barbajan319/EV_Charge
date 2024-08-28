import gym
from gym import spaces
import numpy as np

class MultiAgentEVChargingEnv(gym.Env):
    def __init__(self, num_agents, num_charging_stations, max_soc, max_time_slots):
        super(MultiAgentEVChargingEnv, self).__init__()

        self.num_agents = num_agents
        self.charging_stations = charging_stations
        self.max_soc = max_soc
        self.max_time_slots = max_time_slots
        self.electricity_prices = electricity_prices

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # Charge or idle
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_agents, 3))

        # Initialize state variables
        self.current_time_slot = 0
        self.agents_soc = np.random.uniform(0, self.max_soc, size=(self.num_agents,))
        self.agents_charging_status = np.zeros((self.num_agents,))
        self.charging_stations_occupancy = np.zeros((self.num_charging_stations,))

    def reset(self):
        # Reset environment state
        self.current_time_slot = 0
        self.agents_soc = np.random.uniform(0, self.max_soc, size=(self.num_agents,))
        self.agents_charging_status = np.zeros((self.num_agents,))
        self.charging_stations_occupancy = np.zeros((self.num_charging_stations,))

        # Return initial observation
        return self._get_observation()

    def step(self, actions):

        for agent_id, action in enumerate(actions):
            if action == 1 and self.charging_stations_occupancy[agent_id] == 0:
                self.agents_charging_status[agent_id] = 1
            else:
                self.agents_charging_status[agent_id] = 0

        # Update environment state
        self.current_time_slot += 1
        self.agents_soc += self.agents_charging_status  # Simple charging model (SOC increases when charging)

        # Calculate rewards
        rewards = self._calculate_rewards()

        # Check if the episode is done
        done = self.current_time_slot >= self.max_time_slots

        # Return next observation, rewards, and episode completion flag
        return self._get_observation(), rewards, done, {}

    def _calculate_rewards(self):
        # Simple reward: 1 if agents are charging, 0 otherwise
        rewards = np.sum(self.agents_charging_status)
        return rewards

    def _get_observation(self):
        # Current SOC, time, and charging status for each agent
        observation = np.vstack([self.agents_soc, np.full_like(self.agents_soc, self.current_time_slot),
                                 self.agents_charging_status]).T
        return observation

# # Example usage
# env = MultiAgentEVChargingEnv(num_agents=3, num_charging_stations=3, max_soc=100, max_time_slots=10)
# observation = env.reset()

# for _ in range(10):
#     actions = [env.action_space.sample() for _ in range(env.num_agents)]  # Random actions
#     observation, rewards, done, _ = env.step(actions)
#     print("Observation:", observation)
#     print("Rewards:", rewards)
#     if done:
#         break
