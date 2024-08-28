import random 
import numpy as np
import gymnasium as gym
import panda_gym
import torch as T
from agent import Agent
from episode import EpisodeWorker
from HER import HER
from mpi4py import MPI
from wrappers import FlattenDictWrapper
import os
from tqdm import tqdm
import time 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def train(agent, worker, memory):
    epochs = 200
    cycle_length = 20
    num_cycles = 50 
    n_updates = 20
    n_tests = 10
    success = 0
    best_score = 0
        
    for epoch in range(epochs):
        for cycle in range(num_cycles):
            score_history, success_history = [], []
            for i in range(cycle_length):
                score, success = worker.play_episode()
                score_history.append(score)
                success_history.append(success)
            # if MPI.COMM_WORLD.Get_rank()==0:
            #     cycle_avg_score = np.mean(score_history)
            #     cycle_avg_success = np.mean(success_history)
                
                # print('Epoch: {} Cycle: {}, Training Avg Score {:.1f}'
                #       'Training Avg Success: {:.3f}'.
                #       format(epoch,cycle,cycle_avg_score,cycle_avg_success))
                
            if memory.ready():
                for _ in range(n_updates):
                    memories = memory.sample_memory()
                    agent.learn(memories)
                agent.update_network_parameters()
                
        score_history, success_history = [], []
        for episode in range(n_tests):
            score, success = worker.play_episode(evaluate = True)
            success_history.append(success)
            score_history.append(score)
        avg_success = np.mean(success_history)
        avg_score = np.mean(score_history)
        # global_success = MPI.COMM_WORLD.allreduce(avg_success, op=MPI.SUM)
        # global_score = MPI.COMM_WORLD.allreduce(avg_score, op=MPI.SUM)
        # eval_score = global_score/MPI.COMM_WORLD.Get_size()
        # eval_success = global_success/MPI.COMM_WORLD.Get_size()
        if avg_success > best_score:
            agent.save_models()
            best_score = avg_success
            print("model saved, success rate: ", best_score)
        print('Epoch: {} Testing Agent. Avg Score: {:.1f}, Avg Success: {:.3f}'.format(epoch, avg_score, avg_success))
        
        


        
def main():

    from EVChargeEnv import EVCharge
    env = EVCharge()
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    env.reset()
    
    batch_size = 256
    max_size = 10000
    obs_shape = env.observation_space['observation'].shape[0]
    # print("obs", env.observation_space['observation'], env.observation_space)
    # print("acts", env.action_space)
    goal_shape = env.observation_space['achieved_goal'].shape[0]
    input_shape = obs_shape
    memory = HER(max_mem=max_size, input_shape=input_shape, n_actions=1, 
                 batch_size=batch_size, goal_shape=goal_shape, strategy='future',
                 reward_fn= env.compute_reward)
    
    input_shape = obs_shape + goal_shape
    # print("input_shape", input_shape)
    agent = Agent(alpha=0.0001, beta=0.0001, action_space = env.action_space,
                  input_dims=input_shape, tau = 0.05, gamma = 0.98,
                  fc1_dims=128, fc2_dims=128, fc3_dims=128,
                  n_actions=env.action_space.shape[0], explore=0.3, obs_shape=obs_shape,
                  goal_shape=goal_shape, action_noise=0.2)
    
    ep_worker = EpisodeWorker(env, agent, memory)
    print("begin training")
    train(agent, ep_worker, memory)
    
if __name__ == '__main__':        
    num_cores = 4 
    results = Parallel(n_jobs = num_cores)(
        delayed(main())
    )
    # from EVChargeEnv import EVCharge
    # env = EVCharge()
    # seed = 65
    # random.seed(seed)
    # np.random.seed(seed)
    # T.manual_seed(seed)
    # env.reset()
    # timesteps = env.max_steps
    # obs_shape = env.observation_space['observation'].shape[0]
    # # print("obs", env.observation_space['observation'], env.observation_space)
    # # print("acts", env.action_space)
    # goal_shape = env.observation_space['achieved_goal'].shape[0]
    # input_shape = obs_shape
    # input_shape = obs_shape + goal_shape
    # agent = Agent(alpha=0.001, beta=0.0001, action_space = env.action_space,
    #               input_dims=input_shape, tau = 0.05, gamma = 0.98,
    #               fc1_dims=128, fc2_dims=128, fc3_dims=128,
    #               n_actions=env.action_space.shape[0], explore=0.3, obs_shape=obs_shape,
    #               goal_shape=goal_shape, action_noise=0.2)
    
    # OB = env.observation_space['observation'].shape[0]
    # A = env.observation_space['achieved_goal'].shape[0]
    # D = env.observation_space['desired_goal'].shape[0]
    # ob = slice(0, OB)
    # ag = slice(OB, OB + A)
    # dg = slice(OB + A, OB + A + D)
    
    # all_rates = []
    
    # agent.load_models()
    # for i in range(10):
    #     obs = env.reset()
    #     done = False
    #     battery_level = env._battery_level
    #     target_level = env._desired_goal
    #     departure_time = env._departure_times
    #     desired_goal = obs[dg]
    #     achieved_goal = obs[ag]
    #     observation = obs[ob]
        
    #     achieved_goal = agent.goal_stats.normalize_observation(achieved_goal)
    #     desired_goal = agent.goal_stats.normalize_observation(desired_goal)
    #     observation = agent.obs_stats.normalize_observation(observation)
    #     print('Episode: {} Battery: {:.1f}, target: {:.1f}, departure: {}'.format(i, battery_level, target_level, departure_time))
        
    #     episode_rates = []
    #     episode_soc = [battery_level]
    #     while not done:
    #         action = agent.choose_action(np.concatenate([observation, desired_goal]), True)
    #         rate = action * 350 if action > 0 else 0
    #         episode_rates.append(rate)
    #         observation_, reward, done, info = env.step(action)
            
    #         achieved_goal_new = observation_[ag]
    #         observation_ = observation_[ob]
    #         episode_soc.append(env._battery_level)
            
    #         observation_ = agent.obs_stats.normalize_observation(observation_)
    #         observation = observation_
    #         env.render()
        
    #     plt.figure()
    #     plt.plot(episode_soc, label= 'SOC')
    #     plt.plot(episode_rates, linestyle = '--', label = f'Rate {episode_rates[0]}')
    #     plt.axhline(y=target_level, color='r', linestyle='-', label='Target SOC')
    #     plt.axhline(y= target_level + env.buffer, color = 'black',linestyle='-', label='Buffer')
    #     plt.axvline(x=departure_time, color='g', linestyle='-', label='Departure Time')
    #     plt.xlabel('Timestep')
    #     plt.ylabel('State of Charge (SOC)')
    #     plt.title(f'SOC over Timesteps for Episode {i+1}')
    #     plt.legend()
    #     plt.savefig(f'episode_{i+1}_soc_plot.png')
    #     plt.close()
            
            
            
            
            
        
        
    
    
    