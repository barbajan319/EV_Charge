from envSINGLE import SingleAgentEVCharge_V5
from MADDPG_V6 import MultiAgentReplayBuffer, CriticNetwork, ActorNetwork, Agent, MADDPG, obs_dict_to_state_vector
import math
import numpy as np 
import matplotlib.pyplot as plt
import time 
from joblib import Parallel, delayed
import os
import tqdm

# Define a function to run simulations for a given set of parameters
def run_simulation(alpha, beta, gamma, n_agents):
    
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
    
    print(f"Simulation starting with {n_agents} agents")
    env = SingleAgentEVCharge_V5(num_agents = n_agents, alpha = alpha, beta = beta, gamma = gamma)
    obs = env.reset()
    n_agents = env.num_agents
    agent_names = env.agents
    actor_dims = [3] * n_agents
    overcharge_times = generate_evs(n_agents, 1)
    overcharge_mean = generate_evs(n_agents,1)
    charging_rate_ = generate_evs(n_agents,1)
    undercharge = generate_evs(n_agents,1)
    complete_charge = []
    charge_att = generate_evs(n_agents,1)
    gap_time = generate_evs(n_agents,1)

    critic_dims = sum(actor_dims)
    acts = env.action_space(env.agents[0])
    n_actions = 1
    maddppg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, env,
                            alpha=0.001, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau = 0.01, chkpt_dir='C:/Users/luisb/EVCHARGING/tmp/maddpg_tst')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=10)


    PRINT_INTERVAL = 100
    N_EPS = 500
    total_steps = 0
    score_history = generate_evs(n_agents,1)
    evaluate = False
    best_score = generate_evs(n_agents,2)
    truncations = False
    avg_score =generate_evs(n_agents,3)
    task_completion = []
    indv_completion = generate_evs(n_agents,3)
    if evaluate:
        maddppg_agents.load_checkpoint()
    for i in range(N_EPS):
        obs = env.reset()
        score = generate_evs(n_agents,3)
        terminations = env.terminations
        truncations = env.truncations
        episode_step = 0
        while not all(terminations.values()) and not truncations:
            # if evaluate:
            #     # env.render()
            actions = maddppg_agents.choose_action(obs)
            obs_, rewards, terminations,truncations, _, tc = env.step(actions)

        
            state = obs_dict_to_state_vector(obs, n_agents)
            state_ = obs_dict_to_state_vector(obs_, n_agents)
        
            terminations_list = list(terminations.values())
            memory.store_transition(obs, state, actions, rewards, obs_, state_, terminations_list)
            obs = obs_
        
            total_steps  += 1
            episode_step += 1


        for agent in rewards.keys():
            score[agent] = rewards[agent]
            score_history[agent].append(score[agent])
            avg_score[agent] = np.mean(score_history[agent][-100:])

        maddppg_agents.update_epsilon()
        if not evaluate:
            if memory.ready() and i % 100 == 0:
                maddppg_agents.learn(memory, i)
                for agent in maddppg_agents.agents:
                    if avg_score[agent.agent_name] > best_score[agent.agent_name]:
                        maddppg_agents.save_checkpoint(agent)
                        best_score[agent.agent_name] = avg_score[agent.agent_name]
                        
        

        
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f"Episode :, {i}, Average Score: {avg_score} Best Score: {best_score}")
            print(f"Rewards:  {rewards}\n")

        if evaluate:
            task_completion.append(tc)
            indv_tsk = env.indv_task
            for agent in agent_names:
                if indv_tsk[agent]:
                    indv_completion[agent] += 1
                    
        #Get Metrics
        for EV in agent_names:
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
    # Plot data from overcharge_times

    for agent in maddppg_agents.agents:
        maddppg_agents.save_checkpoint(agent)
    
    
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
    
alphas = 1#[0.1, 0.5, 1, 5,10]
betas= 0.01 #[0.001, 0.1, 0.5, 1, 5]
gammas= 200#[10, 20, 30, 50, 100, 200]
n_agents = [3]


# Run simulations in parallel for all parameter combinations
num_cores = 1  # Number of CPU cores to utilize
results = tqdm(Parallel(n_jobs=num_cores)(
    delayed(run_simulation)(alphas, betas, gammas, agent)
    for agent in  n_agents
))
