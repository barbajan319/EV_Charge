import numpy as np

# from env5 import MultiAgentEVCharge_V5
from ENV_1 import EV_CHARGE
import torch as T
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ReLU
import os
import torch.optim as optim
import time
import math
import tkinter as tk
import random
import shutil
import torch.nn.utils as nn_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import csv
import time
from joblib import Parallel, delayed

T.autograd.set_detect_anomaly(True)


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.25):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = T.ones(self.size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * T.randn(len(x))
        self.state = x + dx
        return self.state


class MultiAgentReplayBuffer:
    def __init__(
        self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size
    ):
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
        """
        Initialize memory for actor states and actions for each agent.
        """
        # Action memory consists of the action taken by each agent in that particular time step
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i]))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i]))
            )
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, rewards, raw_obs_, state_, done):

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
        self.reward_memory[index] = [reward for reward in rewards.values()]
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

        return (
            actor_states,
            states,
            actions,
            rewards,
            actor_new_states,
            states_,
            terminal,
        )

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False


class CriticNetwork(nn.Module):
    def __init__(
        self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir
    ):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(
        self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir
    ):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # first layer takes only each agents observations
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.continous_output = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # Optional GPU Usage:
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        cont_out = self.continous_output(x)
        charge_rate = T.tanh(cont_out)
        return charge_rate

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class Agent:
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_actions,
        n_agents,
        agent_idx,
        chkpt_dir,
        alpha,
        beta,
        fc1,
        fc2,
        gamma,
        tau,
    ):

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = agent_idx
        self.n_agents = n_agents
        self.epsilon = 1
        self.epsilon = T.tensor(self.epsilon, dtype=T.float32)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        self.actor = ActorNetwork(
            alpha,
            actor_dims,
            fc1,
            fc2,
            n_actions,
            name=str(self.agent_name) + "_actor",
            chkpt_dir=chkpt_dir,
        )

        self.critic = CriticNetwork(
            beta,
            critic_dims,
            fc1,
            fc2,
            n_agents,
            n_actions,
            name=str(self.agent_name) + "_critic",
            chkpt_dir=chkpt_dir,
        )

        self.target_actor = ActorNetwork(
            alpha,
            actor_dims,
            fc1,
            fc2,
            n_actions,
            name=str(self.agent_name) + "_target_actor",
            chkpt_dir=chkpt_dir,
        )

        self.target_critic = CriticNetwork(
            beta,
            critic_dims,
            fc1,
            fc2,
            n_agents,
            n_actions,
            name=str(self.agent_name) + "_target_critic",
            chkpt_dir=chkpt_dir,
        )

        self.update_network_parameters(tau=1)
        self.noise = OrnsteinUhlenbeckNoise(size=n_actions)

    def update_network_parameters(self, tau=None):
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
        actor_state_dict = {k: v for k, v in actor_state_dict.items() if "bn" not in k}
        # Perform soft parameter updates by blending the actor network's parameters into the target actor network
        for name in actor_state_dict:
            # Blend the current actor's parameter with the corresponding target actor's parameter
            # using a smoothing factor (tau) to control the degree of blending
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_actor.load_state_dict(actor_state_dict, strict=False)
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_state_dict[name].clone()
            )
        self.target_critic.load_state_dict(critic_state_dict, strict=False)

    def choose_action(self, observation):
        soc = T.tensor(observation["SOC"], dtype=T.float32)
        target_soc = T.tensor(observation["Target_SOC"], dtype=T.float32)
        time_left_depot = T.tensor(observation["Time_left_Depot"], dtype=T.float32)
        safety_buffer = T.tensor(observation["Safety_buffer"], dtype=T.float32)
        state = T.cat((soc, target_soc, time_left_depot, safety_buffer), dim=-1)
        state = state.to(self.actor.device)
        charge_rate = self.actor.forward(state)

        if random.random() < self.epsilon:
            noise = self.noise.noise() * self.epsilon
            charge_rate += T.tensor(noise, dtype=T.float32).to(self.actor.device)

        return T.clamp(charge_rate, -1.0, 1.0).detach().numpy()[0]

    def explore_with_noise(self, action):
        noise = T.normal(-1, 1, size=action.shape)
        noisy_action = action + self.epsilon * noise
        return T.clamp(noisy_action, -1.0, 1.0)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.epsilon < 0.001:
            print("\n")
            print("STOPPED NOISE")
            print("\n")

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
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_agents,
        n_actions,
        env,
        alpha,
        beta,
        fc1,
        fc2,
        gamma,
        tau,
        chkpt_dir="tmp/maddpg/",
    ):

        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.fc1 = fc1
        self.fc2 = fc2
        self.gamma = gamma
        self.tau = tau

        for agent_idx in range(self.n_agents):
            self.agents.append(
                Agent(
                    actor_dims[agent_idx],
                    critic_dims,
                    n_actions,
                    n_agents,
                    agent_idx,
                    chkpt_dir,
                    alpha,
                    beta,
                    fc1,
                    fc2,
                    gamma,
                    tau,
                )
            )

    def save_checkpoint(self, agent):
        print("....saving checkpoint.....", agent.agent_name)
        agent.save_models()

    def load_checkpoint(self):
        print("....loading checkpoint.....")
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, obs):
        actions = []  # might need to update this to a dictionary
        for agent_idx, agent in enumerate(self.agents):
            actions.append(agent.choose_action(obs[agent_idx]))
        return actions

    def update_epsilon(self):
        for agent in self.agents:
            agent.update_epsilon()

    def reset_epsilon(self):
        for agent in self.agents:
            agent.epsilon = 0.4

    def learn(self, memory, episode):
        if not memory.ready():
            return

        (
            actor_states,
            states,
            actions,
            rewards,
            actor_new_states,
            states_,
            terminations,
        ) = memory.sample_buffer()
        device = self.agents[0].actor.device
        states = T.tensor(states, dtype=T.float32).to(device)
        actions_np = np.array(
            actions, dtype=np.float32
        )  # Convert the list of NumPy arrays to a single NumPy array
        actions = T.tensor(actions_np, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        states_ = T.tensor(states_, dtype=T.float32).to(device)
        terminations = T.tensor(terminations).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            # estimate action values for the next state according to actor network
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float32).to(
                device
            )
            new_charge_rate = agent.target_actor.forward(new_states)
            all_agents_new_actions.append((new_charge_rate))

            # Action for current state from actor network
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float32).to(device)
            charge_rate = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append((charge_rate))
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        critic_value = {}
        critic_value_ = {}
        critic_loss = {}
        actor_loss = {}
        target = {}
        for agent_idx, agent in enumerate(self.agents):
            # get the states and new action for the target critic network and flatten them.
            # critic values with target critic
            # One-step lookahead TD-error:
            critic_value_[agent_idx] = agent.target_critic.forward(
                states_, new_actions
            ).flatten()
            # ensure that terminal states are not include in future rewards
            critic_value_[agent_idx][terminations[:, 0]] = 0.0
            # critic values using the local critic
            # network, how good the action actually was
            critic_value[agent_idx] = agent.critic.forward(
                states, old_actions
            ).flatten()
            mean_rewards = T.mean(rewards[:, agent_idx])
            std_rewards = T.std(rewards[:, agent_idx])
            normalized_rewards = (rewards[:, agent_idx] - mean_rewards) / (
                std_rewards + 1e-8
            )
            # target[agent_idx] = normalized_rewards + agent.gamma*critic_value_[agent_idx]
            target[agent_idx] = (
                rewards[:, agent_idx] + agent.gamma * critic_value_[agent_idx]
            )
            # calculate the loss of the current critic value
            critic_loss[agent_idx] = F.mse_loss(
                target[agent_idx], critic_value[agent_idx]
            )
            agent.critic.optimizer.zero_grad()
            critic_loss[agent_idx].backward(retain_graph=True)

            actor_loss[agent_idx] = agent.critic.forward(states, mu).flatten()
            actor_loss[agent_idx] = -T.mean(actor_loss[agent_idx])
            agent.actor.optimizer.zero_grad()
            actor_loss[agent_idx].backward(retain_graph=True)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.step()
            agent.critic.optimizer.step()
            agent.update_network_parameters()


def obs_list_to_state_vector(observation):
    state = np.array([])
    for (
        outer,
        inner,
    ) in observation.items():
        for inner_key, value in inner.items():
            state = np.concatenate([state, value])
    return state


if __name__ == "__main__":

    def run_simulation(alpha, beta, gamma, batch_size, episodes, fc1, fc2):
        def generate_evs(num_evs, flag):
            ev_dict = {}
            for i in range(num_evs):
                if flag == 1:
                    ev_dict[i] = []
                elif flag == 2:
                    ev_dict[i] = -math.inf

                elif flag == 3:
                    ev_dict[i] = 0
            return ev_dict

        env = EV_CHARGE(num_agents=2, alpha=alpha, beta=beta, gamma=gamma)

        n_agents = env.num_agents
        overcharge_times = generate_evs(n_agents, 1)
        overcharge_mean = generate_evs(n_agents, 1)
        charging_rate_ = generate_evs(n_agents, 1)
        undercharge = generate_evs(n_agents, 1)
        complete_charge = []
        charge_att = generate_evs(n_agents, 1)
        gap_time = generate_evs(n_agents, 1)
        possible_agents = env.possible_agents
        actor_dims = [4, 4]
        critic_dims = sum(actor_dims)
        fc1 = fc1
        fc2 = fc2

        n_actions = 1
        maddppg_agents = MADDPG(
            actor_dims,
            critic_dims,
            n_agents,
            n_actions,
            env,
            alpha=0.001,
            beta=0.0001,
            fc1=fc1,
            fc2=fc2,
            gamma=0.99,
            tau=0.01,
            chkpt_dir="C:/Users/luisb/EVCHARGING/tmp/maddpg_tst",
        )

        memory = MultiAgentReplayBuffer(
            50000, critic_dims, actor_dims, n_actions, n_agents, batch_size=batch_size
        )

        PRINT_INTERVAL = 100
        N_EPS = episodes
        total_steps = 0
        score_history = generate_evs(n_agents, 1)
        evaluate = False
        best_score = generate_evs(n_agents, 2)
        truncations = False
        avg_score = generate_evs(n_agents, 3)
        task_completion = []
        indv_completion = generate_evs(n_agents, 3)
        final_ep_SOC = generate_evs(n_agents, 1)
        final_ep_rate = generate_evs(n_agents, 1)
        with open(
            f"excel/agent_mapping_{batch_size}_{alpha}_{beta}_{gamma}.csv",
            "w",
            newline="",
        ) as csvfile:
            fieldnames = [
                "Episode",
                "Agent",
                "Timestep",
                "SOC",
                "Target_SOC",
                "Time_left_Depot",
                "Action",
                "Reward",
                "terminated",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            if evaluate:
                maddppg_agents.load_checkpoint()
                maddppg_agents.agents[0].epsilon = 0
                maddppg_agents.agents[1].epsilon = 0
            for i in range(N_EPS):
                obs = env.reset()
                score = generate_evs(n_agents, 3)
                terminations = env.terminations
                truncations = env.truncations
                episode_step = 0
                while not all(terminations.values()) and not truncations:
                    # if evaluate:
                    #     # env.render()
                    actions = maddppg_agents.choose_action(obs)
                    obs_, rewards, terminations, truncations, _, tc = env.step(actions)

                    state = obs_list_to_state_vector(obs)
                    state_ = obs_list_to_state_vector(obs_)
                    terminations_list = list(terminations.values())
                    memory.store_transition(
                        obs, state, actions, rewards, obs_, state_, terminations_list
                    )
                    obs = obs_
                    total_steps += 1
                    episode_step += 1
                    for agent in possible_agents:
                        writer.writerow(
                            {
                                "Episode": i,
                                "Agent": agent,
                                "Timestep": env.timestep,
                                "SOC": env.SOC[agent],
                                "Target_SOC": env.target_SOC[agent],
                                "Time_left_Depot": env.departure_times[agent]
                                - env.timestep,
                                "Action": env.agent_actions[agent],
                                "Reward": env.rewards[agent],
                                "terminated": env.terminations[agent],
                            }
                        )

                    if i % (N_EPS - 1) == 0 and i > 0:
                        print("o", i, N_EPS, (i % N_EPS - 1))
                        for agent in possible_agents:
                            final_ep_SOC[agent].append(env.SOC[agent])
                            final_ep_rate[agent].append(env.agent_actions[agent])

                for agent in rewards.keys():
                    score[agent] = rewards[agent]
                    score_history[agent].append(score[agent])
                    avg_score[agent] = np.mean(score_history[agent][-50:])

                maddppg_agents.update_epsilon()
                if not evaluate:
                    if memory.ready() and i % 10 == 0:
                        maddppg_agents.learn(memory, i)
                        for agent in maddppg_agents.agents:
                            if (
                                avg_score[agent.agent_name]
                                > best_score[agent.agent_name]
                            ):
                                maddppg_agents.save_checkpoint(agent)
                                best_score[agent.agent_name] = avg_score[
                                    agent.agent_name
                                ]

                if i % PRINT_INTERVAL == 0 and i > 0:
                    print(
                        f"Episode :, {i}, Average Score: {avg_score} Best Score: {best_score}"
                    )
                    print(f"Rewards:  {rewards}\n")

                # if evaluate:
                #     task_completion.append(tc)
                #     indv_tsk = env.indv_task
                #     for agent in agent_names:
                #         if indv_tsk[agent]:
                #             indv_completion[agent] += 1

                # if i % 2500 == 0 and i < 5500:
                #     maddppg_agents.reset_epsilon()

                # Get Metrics
                for EV in possible_agents:
                    # Times agents overcharged their battery in an episode
                    overcharge_times[EV].append(
                        env.contributions[EV]["overcharge_bat"][0]
                    )
                    # The average kWh the EV overcharged

                    avg = (
                        sum(env.contributions[EV]["overcharge_bat"][1])
                        / env.contributions[EV]["overcharge_bat"][0]
                        if env.contributions[EV]["overcharge_bat"][0] != 0
                        else 0
                    )
                    overcharge_mean[EV].append(avg)
                    # Times EV attempted to charge while it had already reached target SOC
                    avg_rate = (
                        env.contributions[EV]["charging_rate"][1]
                        / env.contributions[EV]["charging_rate"][0]
                    )
                    charging_rate_[EV].append(avg_rate)
                    # kWh short from completing goal
                    undercharge[EV].append(env.contributions[EV]["undercharge"][1])
                    # How many times each EV entered the charging station
                    charge_att[EV].append(env.charging_attempt[EV])
                    # Timesteps taken to close the gap
                    gap_time[EV].append(env.contributions[EV]["gap"][0])
                # How many Evs completed charging
                complete_charge.append(env.complete_charge)
            # Create two subplots: one for overcharge_times and one for overcharge_att
        # Plot data from overcharge_times

        # for agent in maddppg_agents.agents:
        #     maddppg_agents.save_checkpoint(agent)

        folder_name = f"Test_2/{fc1}/Batch_{batch_size}/{alpha}_{beta}_{gamma}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        window_size = 1
        plt.figure(figsize=(6, 4))
        for ev, values in overcharge_times.items():
            moving_avg = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(np.arange(window_size - 1, len(values)), moving_avg, label=ev)
        plt.title(f"Overcharge Times [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.legend()
        plt.savefig(
            os.path.join(
                folder_name,
                f"Overcharge_times_{alpha}_{beta}_{gamma}_ {batch_size}.png",
            )
        )
        plt.close()

        # Plot data from overcharge_mean
        plt.figure(figsize=(6, 4))
        for ev, values in overcharge_mean.items():
            moving_avg = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(np.arange(window_size - 1, len(values)), moving_avg, label=ev)
        plt.title(f"Overcharge Mean [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Index")
        plt.ylabel("kW")
        plt.legend()
        plt.savefig(
            os.path.join(
                folder_name, f"Overcharge_mean_{alpha}_{beta}_{gamma}_ {batch_size}.png"
            )
        )
        plt.close()

        # Plot data from undercharge
        plt.figure(figsize=(6, 4))
        for ev, values in undercharge.items():
            moving_avg = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(np.arange(window_size - 1, len(values)), moving_avg, label=ev)
        plt.title(f"Undercharge [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Index")
        plt.ylabel("kW")
        plt.legend()
        plt.savefig(
            os.path.join(
                folder_name,
                f"Undercharge_mean_{alpha}_{beta}_{gamma}_ {batch_size}.png",
            )
        )
        plt.close()

        # Plot data from charge_att
        plt.figure(figsize=(6, 4))
        for ev, values in charging_rate_.items():
            moving_avg = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(np.arange(window_size - 1, len(values)), moving_avg, label=ev)
        plt.title(f"Charge Rate [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Index")
        plt.ylabel("kW")
        plt.savefig(
            os.path.join(
                folder_name, f"charging_rate_{alpha}_{beta}_{gamma}_ {batch_size}.png"
            )
        )
        plt.close()

        # Plot data from gap_time
        plt.figure(figsize=(6, 4))
        for ev, values in gap_time.items():
            moving_avg = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(np.arange(window_size - 1, len(values)), moving_avg, label=ev)
        plt.title(f"Gap Time [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Index")
        plt.ylabel("Time")
        plt.legend()
        plt.savefig(
            os.path.join(
                folder_name, f"gap_time_{alpha}_{beta}_{gamma}_ {batch_size}.png"
            )
        )
        plt.close()

        plt.figure(figsize=(6, 4))
        for ev, values in final_ep_SOC.items():
            plt.plot(range(len(values)), values, label=str(env.departure_times[ev]))
        for ev, values in final_ep_rate.items():
            plt.plot(range(len(values)), values, linestyle="dotted")
        plt.axhline(y=env.target_SOC[0], color="blue", linestyle="--")
        plt.axhline(y=env.safety_buffer[0], color="blue", linestyle="-")
        plt.axhline(y=env.target_SOC[1], color="black", linestyle="--")
        plt.axhline(y=env.safety_buffer[1], color="black", linestyle="-")
        plt.title(f"SOC by timestep [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Time Step (15 min interval)")
        plt.ylabel("Battery")
        plt.legend()
        plt.savefig(
            os.path.join(
                folder_name, f"soc_timestep_{alpha}_{beta}_{gamma}_ {batch_size}.png"
            )
        )
        plt.close()

        # Plot data from complete_task
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(complete_charge)), complete_charge)
        plt.xticks(range(len(values)))
        plt.title(f"EVs Completion [a={alpha}, B={beta}, y={gamma}]")
        plt.xlabel("Index")
        plt.ylabel("EVs")
        plt.savefig(
            os.path.join(
                folder_name, f"complete_task_{alpha}_{beta}_{gamma}_ {batch_size}.png"
            )
        )
        plt.close()

    batch_sizes = [512]

    alphas = [1]
    gammas = [20]
    num_cores = 4  # Number of CPU cores to utilize
    results = Parallel(n_jobs=num_cores)(
        delayed(run_simulation)(alpha, 0, gamma, batch, 8000, 32, 64)
        for alpha in alphas
        for gamma in gammas
        for batch in batch_sizes
    )

#     results = Parallel(n_jobs=num_cores)(
#         delayed(run_simulation)(low[0], low[1], low[2], batch, 3000, 4, 8)
#         for low in low
#         for batch in batch_sizes

#  )

#     results = Parallel(n_jobs=num_cores)(
#         delayed(run_simulation)(low[0], low[1], low[2], batch, 3000, 2, 4)
#         for low in low
#         for batch in batch_sizes

#  )
# Run simulations in parallel for all parameter combinations
#     num_cores = 4  # Number of CPU cores to utilize
#     results = Parallel(n_jobs=num_cores)(
#         delayed(run_simulation)(alpha, beta, gamma, batch, 2500, 8, 16)
#         for batch in  batch_sizes
#         for alpha in alphas
#         for beta in betas
#         for gamma in gammas

# )
