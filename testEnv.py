from EVenv import DepotEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.test import api_test

if __name__ == "__main__":
    env = DepotEnv(num_agents=10)
    parallel_api_test(env, num_cycles=100000)