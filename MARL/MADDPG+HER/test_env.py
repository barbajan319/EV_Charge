from EVChargeEnv import EVCharge

if __name__ == '__main__':
    env = EVCharge()
    for _ in range(2):
        done = False
        obs = env.reset()
        print("starting new episode")
        while not done:
            action = env.action_space_sample()
            obs_, reward, done, infor = env.step(action)
            env.render()