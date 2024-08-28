import numpy as np
from MATD3 import Agent
from env import EV_CHARGE
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = EV_CHARGE(num_agents=1, alpha=1, beta=1, gamma=1)
    #env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.01, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=10000, layer1_size=8, layer2_size=16,
                n_actions=env.action_space.shape[0])
    
    n_games = 5000
    best_score = -1000
    score_history = []
    final_ep_SOC = []
    final_ep_rate = []
    total_rewards = []
    #agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            
            score += reward
            observation = observation_
            
            if i % (n_games - 1) == 0 and i > 0:
                final_ep_SOC.append(env.SOC)
                final_ep_rate.append(env.rate)
        total_rewards.append(env.cummulative_rewards[0])
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        agent.learn()
         

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)

    folder_name = f'Test3'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
            
    window_size = 500
    plt.figure(figsize=(6, 4))
    moving_avg = np.convolve(total_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(total_rewards)), moving_avg, label = "l")
    plt.title(f'Total_Rewards')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Total_Rewards.png'))
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(final_ep_SOC)), final_ep_SOC, label='SOC')
    plt.plot(range(len(final_ep_rate)), final_ep_rate, linestyle = 'dotted', label="Rate")
    plt.axhline(y=env.target_SOC, color='r', linestyle='--')
    plt.axhline(y=env.safety_buffer, color='black', linestyle='--')
    plt.title(f'SOC by timestep')
    plt.xlabel('Time Step (15 min interval)')
    plt.ylabel('Battery')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'soc_timestep.png'))
    plt.close()