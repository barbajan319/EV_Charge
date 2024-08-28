import pandas as pd
import numpy as np
import gymnasium as gym
from gym import spaces


class EVChargingEnvironment(gym.Env):
    def __init__(self, num_agents, EV_Capacity, price_data, load_data):
        super(EVChargingEnvironment, self).__init__()
        # Number of agents (households with EVs)
        self.num_agents = num_agents
        # EV capacity constraint
        self.EV_Capacity = EV_Capacity
        self.agents = list(range(num_agents))
        # The maximum charging rate accounts for 20% of the capacity
        # Maximum charging/discharging rate for the EV battery
        self.max_charge_rate = self.EV_Capacity * 0.2
        # The transformer capacity is set to be 3.36 N kWh, where N is the number of households connected to the same transformer.
        # Transformer capacity constraint
        self.transformer_capacity = 3.36 * self.num_agents
        # Hourly electricity prices
        self.price_data = price_data
        # Hourly Household load consumption (Non-EV)
        self.load_data = load_data
        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_agents,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_agents, 24 + 24 + 4), dtype=np.float32
        )
        # Initialize state
        self.state = self.reset()
        self.alpha = 1
        self.beta = 5
        self.gamma = 20
        self.day = 1

    def evaluate(self):
        self.electricity_prices, self.selected_date = self.set_electricity_price_eval(
            self.price_data
        )

        self.residential_loads = self.set_residential_load_eval(
            self.load_data, self.selected_date, self.agents
        )

        self.schedules = {agent: self.generate_schedules() for agent in self.agents}
        self.arrival = {agent: self.schedules[agent][0] for agent in self.agents}
        self.departure = {agent: self.schedules[agent][1] for agent in self.agents}
        self.SOC = {agent: self.schedules[agent][2] for agent in self.agents}
        self.target_SOC = {agent: self.schedules[agent][3] for agent in self.agents}
        self.location = {agent: 1 for agent in self.agents}
        self.non_EV_consumption = self.non_EV_load(
            self.residential_loads, self.agents, self.time_step
        )
        self.state = {agent: self._get_observation(agent) for agent in self.agents}
        self.charging_power = {agent: 0 for agent in self.agents}
        self.evaluation = True

        return self.state

    def reset(self):
        self.evaluation = False
        self.time_step = 0
        # Reset the state of the environment to an initial state
        self.electricity_prices, self.selected_date = self.set_electricity_price(
            self.price_data
        )
        self.residential_loads = self.set_residential_load(
            self.load_data, self.selected_date, self.agents
        )
        # Reset the arrival and departure times
        # Extract the schedules for easier use
        self.schedules = {agent: self.generate_schedules() for agent in self.agents}
        self.arrival_time = {agent: self.schedules[agent][0] for agent in self.agents}
        self.departure_time = {agent: self.schedules[agent][1] for agent in self.agents}
        self.arrival = {agent: self.time_slot(agent)[0] for agent in self.agents}
        self.departure = {agent: self.time_slot(agent)[1] for agent in self.agents}
        self.SOC = {agent: self.schedules[agent][2] for agent in self.agents}
        self.target_SOC = {agent: self.schedules[agent][3] for agent in self.agents}
        self.location = {agent: 0 for agent in self.agents}
        self.non_EV_consumption = self.non_EV_load(
            self.residential_loads, self.agents, self.time_step
        )
        # Extract observations for each agent
        self.state = {agent: self._get_observation(agent) for agent in self.agents}

        self.total_rewards = {agent: 0 for agent in self.agents}
        self.cost_rewards = {agent: 0 for agent in self.agents}
        self.RA_rewards = {agent: 0 for agent in self.agents}
        self.TO_rewards = {agent: 0 for agent in self.agents}

        return self.state

    def _get_observation(self, agent):
        # Combine the electricity prices, residential loads, and EV states into a single observation
        prices = self.electricity_prices["HB_HOUSTON"].values
        electricity_prices = [
            float(price.strip().replace("$", "").replace(",", "")) / 1000
            for price in prices
        ]

        household_loads = self.residential_loads[
            self.residential_loads["household_number"] == agent + 1
        ]
        household_loads = household_loads["energy_kWh"].values

        location = self.location[agent]
        departure = self.departure[agent]
        remaining_energy = self.SOC[agent]
        desired_energy = self.target_SOC[agent]
        obs = np.hstack(
            [
                electricity_prices,
                household_loads,
                location,
                departure,
                remaining_energy,
                desired_energy,
            ]
        )

        return obs

    def get_current_price(self):
        prices = self.electricity_prices["HB_HOUSTON"].values
        electricity_prices = [
            float(price.strip().replace("$", "").replace(",", "")) / 1000
            for price in prices
        ]
        return electricity_prices[-1]

    def step(self, actions):
        # Execute one time step within the environment
        actions = np.clip(actions, -1, 1)
        self.rate = actions * self.max_charge_rate
        # Update the state based on actions
        self.ev_consumption = [0] * self.num_agents
        done = False

        # Example: Add the actions to the EV state (charging/discharging)
        for agent in self.agents:
            battery = self.SOC[agent] * self.EV_Capacity
            ev_rate = self.rate[agent]
            if self.location[agent] == 1:
                new_soc = (
                    np.clip(battery + ev_rate, 0, self.EV_Capacity) / self.EV_Capacity
                )
                self.SOC[agent] = new_soc
                self.ev_consumption[agent] = ev_rate
            else:
                self.rate[agent] = 0

        self.transformer_load = sum(self.non_EV_consumption) + sum(self.ev_consumption)
        if sum(self.ev_consumption) == 0:
            self.transformer_load = 0
        self.current_price = self.get_current_price()
        reward = {
            agent: self._calculate_reward(agent, self.rate[agent], self.current_price)
            for agent in self.agents
        }
        self.time_step += 1
        if self.time_step % 24 == 0:
            self.day += 1

        # change location
        for agent in self.agents:
            if self.evaluation:
                if self.departure[agent] == self.time_step:
                    schedule = self.generate_schedules()
                    self.arrival[agent] = schedule[0] + 24 * (self.day - 1)
                    self.departure[agent] = schedule[1] + 24 * self.day
                    self.SOC[agent] = schedule[2]
                    self.target_SOC[agent] = schedule[3]
                    self.location[agent] = 0

                if self.arrival[agent] == self.time_step:
                    self.location[agent] = 1
            else:
                if (
                    self.arrival[agent] <= self.time_step
                    and self.departure[agent] > self.time_step
                ):
                    self.location[agent] = 1
                else:
                    self.location[agent] = 0

        # change electricity observation
        if self.evaluation:
            self.electricity_prices, _ = self.set_electricity_price_eval(
                self.price_data, current_datetime=self.selected_date
            )
            # change load profiles
            self.residential_loads = self.set_residential_load_eval(
                self.load_data, self.selected_date, self.agents
            )
        else:
            self.electricity_prices, _ = self.set_electricity_price(
                self.price_data, current_datetime=self.selected_date
            )
            # change load profiles
            self.residential_loads = self.set_residential_load(
                self.load_data, self.selected_date, self.agents
            )

        self.non_EV_consumption = self.non_EV_load(
            self.residential_loads, self.agents, self.time_step
        )
        # Update the state
        self.state = {agent: self._get_observation(agent) for agent in self.agents}

        # Check if the episode is done
        #
        if self.time_step == 23:
            done = True

        return self.state, reward, done

    def _calculate_reward(self, agent, rate, price):
        arrival = self.arrival[agent]
        charging_cost_penalty = 0
        range_anxiety_penalty = 0
        transformer_overload_penalty = 0
        if arrival < self.time_step < self.departure[agent]:
            charging_cost_penalty = -price * rate

        if self.time_step == self.departure[agent]:
            range_anxiety_penalty = (
                -max(self.target_SOC[agent] - self.SOC[agent], 0) ** 2
            )

        if abs(self.transformer_load) > self.transformer_capacity:
            transformer_overload_penalty = (
                -(self.ev_consumption[agent] / self.transformer_load)
                * (abs(self.transformer_load) - self.transformer_capacity) ** 2
            )

        else:
            transformer_overload_penalty = 0

        reward = (
            self.alpha * charging_cost_penalty
            + self.beta * range_anxiety_penalty
            + self.gamma * transformer_overload_penalty
        )

        self.total_rewards[agent] += reward
        self.cost_rewards[agent] += self.alpha * charging_cost_penalty
        self.RA_rewards[agent] += self.beta * range_anxiety_penalty
        self.TO_rewards[agent] += +self.gamma * transformer_overload_penalty

        return reward

    def render(self, mode="human"):
        # Render the environment to the screen
        # print(f"State: {self.state}")
        pass

    def close(self):
        pass

    def set_electricity_price(self, prices, current_datetime=None):
        prices["DATE"] = pd.to_datetime(prices["DATE"])
        if current_datetime is None:
            unique_dates = prices["DATE"].dt.date.unique()
            unique_dates = unique_dates[
                ~(
                    pd.to_datetime(unique_dates).is_month_start
                    | (pd.to_datetime(unique_dates).day == 20)
                )
            ]
            selected_date = np.random.choice(unique_dates)
        else:
            selected_date = current_datetime
        end_datetime = pd.to_datetime(f"{selected_date} 13:00:00") + pd.Timedelta(
            hours=self.time_step
        )
        start_datetime = end_datetime - pd.Timedelta(hours=24)

        prices["DATETIME"] = pd.to_datetime(
            prices["DATE"].astype(str) + " " + prices["START_TIME"]
        )
        selected_data = prices[
            (prices["DATETIME"] >= start_datetime) & (prices["DATETIME"] < end_datetime)
        ].copy()
        selected_data = selected_data.sort_values(by="DATETIME")
        return selected_data, selected_date

    def set_residential_load(self, loads, date, agents):
        residents = []
        selected_date = date
        for agent in agents:
            residents.append(agent + 1)
        loads["date"] = pd.to_datetime(loads["date"])
        filtered_df = loads[loads["household_number"].isin(residents)].copy()
        end_datetime = pd.to_datetime(f"{selected_date} 13:00:00") + pd.Timedelta(
            hours=self.time_step
        )
        start_datetime = end_datetime - pd.Timedelta(hours=23)
        filtered_df["hour"] = filtered_df["hour"].astype(str).str.zfill(2) + ":00:00"
        filtered_df.loc[:, "datetime"] = pd.to_datetime(
            filtered_df["date"].dt.strftime("%Y-%m-%d") + " " + filtered_df["hour"]
        )
        filtered_df = filtered_df[
            (filtered_df["datetime"] >= start_datetime)
            & (filtered_df["datetime"] <= end_datetime)
        ]
        return filtered_df

    def generate_schedules(self):
        # Arrival times
        def generate_value(mean, std, lower, upper):
            value = np.random.normal(mean, std)
            return np.clip(value, lower, upper)

        arrival_time = int(generate_value(17, 3.6, 16, 20))
        departure_time = int(generate_value(10, 3.2, 6, 11))
        soc = round(generate_value(0.15, 0.1, 0, 0.3), 3)
        target_soc = round(generate_value(0.925, 0.1, 0.85, 1), 3)
        return arrival_time, departure_time, soc, target_soc

    def non_EV_load(self, residential_loads, agents, timestep):

        consumption = []
        for agent in agents:
            household = agent + 1
            household_df = residential_loads[
                residential_loads["household_number"] == household
            ]
            energy = household_df.iloc[-1]["energy_kWh"]
            consumption.append(energy)

        return consumption

    def time_slot(self, agent):
        departure = self.departure_time[agent] + 11
        arrival = self.arrival_time[agent] - 13
        return arrival, departure

    def set_electricity_price_eval(self, prices, current_datetime=None):
        if current_datetime is None:
            prices["DATE"] = pd.to_datetime(prices["DATE"])
            unique_months = prices["DATE"].dt.to_period("M").unique()
            selected_month = np.random.choice(unique_months)
            month_data = prices[prices["DATE"].dt.to_period("M") == selected_month]
            # Filter to include only dates from the 22nd to the 25th
            if selected_month.strftime("%Y-%m")[-2:] != "02":
                filtered_month_data = month_data[
                    (month_data["DATE"].dt.day >= 22)
                    & (month_data["DATE"].dt.day <= 25)
                ]
            else:
                filtered_month_data = month_data[(month_data["DATE"].dt.day == 22)]
            selected_date = pd.to_datetime(
                np.random.choice(filtered_month_data["DATE"])
            ).date()
        else:
            selected_date = current_datetime
        end_datetime = pd.to_datetime(f"{selected_date} 00:00:00") + pd.Timedelta(
            hours=self.time_step
        )
        start_datetime = end_datetime - pd.Timedelta(hours=24)
        prices["DATETIME"] = pd.to_datetime(
            prices["DATE"].astype(str) + " " + prices["START_TIME"]
        )
        selected_data = prices[
            (prices["DATETIME"] >= start_datetime) & (prices["DATETIME"] < end_datetime)
        ].copy()
        selected_data = selected_data.sort_values(by="DATETIME")

        return selected_data, selected_date

    def set_residential_load_eval(self, loads, date, agents):
        residents = []
        selected_date = date
        for agent in agents:
            residents.append(agent + 1)
        loads["date"] = pd.to_datetime(loads["date"])
        filtered_df = loads[loads["household_number"].isin(residents)].copy()
        end_datetime = pd.to_datetime(f"{selected_date} 00:00:00") + pd.Timedelta(
            hours=self.time_step
        )
        start_datetime = end_datetime - pd.Timedelta(hours=23)
        filtered_df["hour"] = filtered_df["hour"].astype(str).str.zfill(2) + ":00:00"
        filtered_df.loc[:, "datetime"] = pd.to_datetime(
            filtered_df["date"].dt.strftime("%Y-%m-%d") + " " + filtered_df["hour"]
        )
        filtered_df = filtered_df[
            (filtered_df["datetime"] >= start_datetime)
            & (filtered_df["datetime"] <= end_datetime)
        ]
        return filtered_df
