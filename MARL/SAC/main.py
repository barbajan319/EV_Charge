import numpy as np
import matplotlib.pyplot as plt
from sac import Agent
from ENV import EVChargingEnvironment
import pandas as pd
import torch as T
import math
from joblib import Parallel, delayed

if __name__ == "__main__":

    def aug_ob(agent, obs):
        prices = obs[agent][:24]
        load_pred = agents[agent].predict_load(prices)
        if env.location[agent] == 0:
            load_pred = T.tensor(0.0)
        augmented_observation = T.from_numpy(np.array(obs[agent])).float()
        augmented_observation = T.cat((augmented_observation, load_pred), dim=0)
        return prices, augmented_observation

    def consumption(agent, env):
        collective_consumption = (
            env.transformer_load
            - env.ev_consumption[agent]
            - env.non_EV_consumption[agent]
        )
        return collective_consumption

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    window_size = 100
    num_agents = 3
    EV_capacity = 24
    gradient_step = 1
    load_data = pd.read_csv("Household_Loads.csv")
    evaluate = False
    if evaluate:
        price_data = pd.read_csv("Test.csv")
    else:
        price_data = pd.read_csv("Train.csv")

    env = EVChargingEnvironment(
        num_agents=num_agents,
        EV_Capacity=EV_capacity,
        price_data=price_data,
        load_data=load_data,
    )

    agents = {
        agent: Agent(
            input_dims=[env.observation_space.shape[1]],
            env=env,
            n_actions=1,
            agent=agent,
        )
        for agent in range(num_agents)
    }

    def run_simulation(env, agents, window_size, gradient_step):
        episodes = 6000
        best_score = {agent: -math.inf for agent in agents}
        score_history = {agent: [] for agent in agents}
        average_score = {agent: 0 for agent in agents}
        load_checkpoint = False
        total_rewards = {agent: [] for agent in agents}
        cost_rewards = {agent: [] for agent in agents}
        RA_rewards = {agent: [] for agent in agents}
        TO_rewards = {agent: [] for agent in agents}

        if load_checkpoint:
            for agent in agents:
                agent.load_models()

        for i in range(episodes):
            observation = env.reset()
            done = False
            score = {agent: 0 for agent in agents}
            while not done:
                actions = []
                aug_obs = []
                aug_obs_ = []
                for agent in agents:
                    prices, augmented_observation = aug_ob(agent, observation)
                    # From augmente observations choose the action
                    action = agents[agent].choose_action(augmented_observation)
                    actions.append(action)
                    aug_obs.append(augmented_observation)

                observation_, reward, done = env.step(actions)
                for agent in agents:
                    score[agent] += reward[agent]
                    prices_, augmented_observation = aug_ob(agent, observation_)
                    aug_obs_.append(augmented_observation)
                    collective_consumption = consumption(agent, env)
                    agents[agent].rememberD1(prices, collective_consumption)
                    agents[agent].rememberD2(
                        aug_obs[agent].cpu().detach().numpy(),
                        actions[agent],
                        reward[agent],
                        aug_obs_[agent].cpu().detach().numpy(),
                        done,
                    )
                    observation[agent] = observation_[agent]

            if not load_checkpoint:
                for step in range(gradient_step):
                    for agent in agents:
                        agents[agent].learn()

            for agent in agents:
                score_history[agent].append(score[agent])
                average_score[agent] = np.mean(score_history[agent][-100:])

                if agents[agent].is_ready():
                    if average_score[agent] > best_score[agent]:
                        best_score[agent] = average_score[agent]
                        if not load_checkpoint:
                            agents[agent].save_models()
                total_rewards[agent].append(env.total_rewards[agent])
                cost_rewards[agent].append(env.cost_rewards[agent])
                RA_rewards[agent].append(env.RA_rewards[agent])
                TO_rewards[agent].append(env.TO_rewards[agent])

            print(f"Episode: {i}, score: {score}, avg score: {average_score}")

        summed_total = [sum(totals) for totals in zip(*total_rewards.values())]
        summed_cost = [sum(costs) for costs in zip(*cost_rewards.values())]
        summed_RA = [sum(RAs) for RAs in zip(*RA_rewards.values())]
        summed_TO = [sum(TOs) for TOs in zip(*TO_rewards.values())]

        smoothed_total = moving_average(summed_total, window_size)
        smoothed_cost = moving_average(summed_cost, window_size)
        smoothed_RA = moving_average(summed_RA, window_size)
        smoothed_TO = moving_average(summed_TO, window_size)

        training_episodes = np.arange(6000)
        smoothed_episodes = training_episodes[: len(smoothed_total)]

        plt.figure(figsize=(10, 4))
        plt.plot(training_episodes, summed_total, alpha=0.3, color="darkred")
        plt.plot(smoothed_episodes, smoothed_total, label="Total", color="darkred")

        plt.plot(training_episodes, summed_cost, alpha=0.3, color="orange")
        plt.plot(
            smoothed_episodes, smoothed_cost, label="Charging Cost", color="orange"
        )

        plt.plot(training_episodes, summed_RA, alpha=0.3, color="green")
        plt.plot(smoothed_episodes, smoothed_RA, label="Range Anxiety", color="green")

        plt.plot(training_episodes, summed_TO, alpha=0.3, color="purple")
        plt.plot(smoothed_episodes, smoothed_TO, label="Overload", color="purple")

        plt.xlabel("Training episodes")
        plt.ylabel("Episode Reward")
        plt.legend()
        plt.savefig("Training_Performance.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 4))

        for key, values in total_rewards.items():
            smoothed_total = moving_average(values, window_size)
            plt.figure(figsize=(10, 4))
            plt.plot(values, alpha=0.3, color="darkred")
            plt.plot(smoothed_total, label="Total", color="darkred")
            plt.xlabel("Training Episodes")
            plt.ylabel("Rewards")
            legend = plt.legend(title=f"Agent#{key}")
            plt.setp(legend.get_title(), color="red")  # Setting the title color to red
            plt.legend()
            plt.savefig(f"Total_Rewards_Agent_{key}.png")
            plt.show()
            plt.close()

    if not evaluate:
        num_cores = 4  # Number of CPU cores to utilize
        window_size = [100]
        results = Parallel(n_jobs=num_cores)(
            delayed(run_simulation)(env, agents, window_size, gradient_step)
            for window_size in window_size
        )

    else:
        for agent in agents:
            agents[agent].load_models()

        observation = env.evaluate()
        charging_power = {agent: [] for agent in agents}
        electricity_price = []
        soc = {agent: [] for agent in agents}
        real_cons = {agent: [] for agent in agents}
        approx_cons = {agent: [] for agent in agents}
        total_consumption = []
        elec_price = []
        time_step = []

        while env.time_step <= 120:
            actions = []
            aug_obs = []
            aug_obs_ = []
            time_step.append(env.time_step)

            for agent in agents:
                prices, augmented_observation = aug_ob(agent, observation)
                pred_cons = augmented_observation[-1]
                approx_cons[agent].append(pred_cons)

                action = agents[agent].choose_action(augmented_observation)
                actions.append(action)

            observation_, reward, done = env.step(actions)
            observation[agent] = observation_[agent]
            total_consumption.append(env.transformer_load)
            elec_price.append(env.current_price)
            for agent in agents:
                real_consumption = sum(
                    value for key, value in env.ev_consumption.items() if key != agent
                )
                real_cons[agent].append(real_consumption)
                charging_power[agent].append(env.rate[agent])
                if env.location[agent] == 1:
                    soc[agent].append(env.SOC[agent])
                else:
                    soc[agent].append(0)

        # Get a graph for each agent for charging power against electricity price
        for agent in agents:
            fig, ax1 = plt.subplots()
            ax1.bar(time_step, charging_power[agent], color="blue")
            ax1.set_xlabel("Time (h)")
            ax1.set_ylabel("Charging power (kW)", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(time_step, electricity_price, color="red")
            ax2.set_ylabel("Electricity price ($/kWh)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            plt.title(f"EV#{agent + 1}")
            plt.savefig(f"/eval/charging_power_{agent}", format="png")
            plt.show()
            plt.close()

            fig, ax = plt.subplots()

            # Plotting SoC (bar chart)
            ax.bar(time_step, soc[agent], color="brown")

            # Customize labels and ticks
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("SoC")
            ax.set_xticks(np.arange(0, 121, 24))
            ax.set_yticks(np.linspace(0, 1, 6))

            # Add alternating background shading
            for i in range(0, 121, 48):
                ax.axvspan(i, i + 24, facecolor="lightgray", alpha=0.5)

            # Save the plot to a file
            plt.savefig(f"/eval/SoC_{agent}", format="png")
            # Show the plot
            plt.show()
            plt.close()

            fig, ax = plt.subplots()
            ax.bar(time_step, real_cons[agent], color="grey", label="Real Consumption")
            ax1.set_xlabel("Time (h)")
            ax1.set_ylabel("Load (kW)", color="grey")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(
                time_step,
                approx_cons[agent],
                color="orange",
                label="Approx Consumption",
            )
            ax2.tick_params(axis="y", labelcolor="orange")
            plt.title(f"EV#{agent + 1}")
            plt.savefig(f"/eval/power_consumption_{agent}", format="png")
            plt.show()
            plt.close()

        fig, ax = plt.subplots()
        ax.bar(time_step, total_consumption, width=0.4, color="red")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Load (kW)")
        ax.set_xticks(np.arange(0, 121, 24))
        ax.set_yticks(np.linspace(-10, 15, 6))

        cap = env.transformer_capacity
        ax.axhline(y=cap, color="red", linestyle="--", linewidth=1)
        ax.axhline(y=-cap, color="red", linestyle="--", linewidth=1)

        ax.annotate(
            str(cap),
            xy=(1, cap),
            xytext=(5, 12),
            arrowprops=dict(facecolor="red", shrink=0.05),
            color="red",
        )
        ax.annotate(
            str(cap),
            xy=(1, -cap),
            xytext=(5, -8),
            arrowprops=dict(facecolor="red", shrink=0.05),
            color="red",
        )

        plt.savefig(f"/eval/Total_power_consumption", format="png")
        plt.show()
        plt.close()


"""
    for each episode do:
        for each timestep do:
            Get actions for each agent
            Execute actions and get reward and new state
            calculate te real energy consumption
            store the price and real energy consumption in D1
            store transition in D2
        for each gradient step do:
            for each agent do:
                update the weights of the collective policy model
                update the buffer of D2 with current collective policy model
                update weights of actor network
                update weights of critic network
                clear buffer D1    
    """
# Collective Policy
# prices = observation[agent][:24]
# load_prediction = agents[agent].predict_load(prices)
# augmented_observation = T.from_numpy(
#     np.array(observation[agent])
# ).float()
# augmented_observation = T.cat(
#     (augmented_observation, load_prediction), dim=0
# )

# prices_ = observation_[agent][:24]
# load_pred_ = agents[agent].predict_load(prices_)
# augmented_obs_ = T.from_numpy(np.array(observation_[agent])).float()
# augmented_observation = T.cat((augmented_obs_, load_pred_), dim=0)
