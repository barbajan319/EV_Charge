import argparse

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from EVenv import DepotEnv
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo import ParallelEnv
from typing import Dict, Any


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class MultiAgentDepotEnv(MultiAgentEnv):
    def __init__(
        self,
        config: Dict[Any, Any] = None,
        num_agents: int = 1
    ):
        super().__init__()
        self.env = DepotEnv(num_agents)
        self.config = config or {}
        self.num_agents = num_agents
        self.agents = [f"EV_{i}" for i in range(num_agents)]

    def reset(self):
        observations, infos = self.env.reset()
        return {agent: observations for agent in self.agents}, {agent: info for agent, info in infos.items()}

    def step(self, action_dict):
        actions = [action_dict[agent] for agent in self.agents]
        observations, rewards, dones, truncations, infos = self.env.step(actions)
        return (
            {agent: observations for agent, observations in zip(self.agents, observations)},
            {agent: reward for agent, reward in zip(self.agents, rewards)},
            {agent: done for agent, done in zip(self.agents, dones)},
            {agent: truncation for agent, truncation in zip(self.agents, truncations)},
            {agent: info for agent, info in zip(self.agents, infos)}
        )

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

config = PPOConfig().training(gamma=0.9, lr=0.0).environment(DepotEnv(num_agents=4)).resources(num_gpus=0).rollouts(num_rollout_workers = 2)
ppotrainer = config.build()
# class MultiAgentDepot(MultiAgentEnv):
    
#     def __init__(self, config: Dict[Any, Any], num_agents: int = 4):
#         super().__init__()
#         if env is None: 
#             self.env = DepotEnv(num_agents=5)
            
#         else:
#             self.env = env
            
#         self.env.reset()
#         self._skip_env_checking = True
        
        
#         self.config = config
        
#         self.observation_space = self.env.observation_space(self.env.agents[0]) # All agents share the same observation space
        
#         self.action_space = self.env.action_space(self.env.agents[0])
        
#         assert all(
#             self.env.observation_space(agent) == self.observation_space
#             for agent in self.env.agents
#         ),(
#         )
#         self._agent_ids = set(self.env.agents)

# from ray.tune.registry import register_env
# from ray.rllib.env import PettingZooEnv

# # Define how to make the environment. This way takes an optional environment config, num_floors
# env_creator = lambda config: DepotEnv(num_floors=config.get("num_agents", 4))

# # Register that way to make the environment under an rllib name
# register_env("depot_env", lambda config: ParallelPettingZooEnv(env_creator(config)))

# ray.rllib.utils.check_env(["depot_env"])

# config = (
#     PPOConfig()
#     .environment("depotenv")
#     .framework("torch")
#     .rollouts(num_rollout_workers=0)
# )

# algo = config.build()
# algo.train()

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--num-gpus",
#     type=int,
#     default=0,
#     help="Number of GPUs to use for training.",
# )
# parser.add_argument(
#     "--as-test",
#     action="store_true",
#     help="Whether this script should be run as a test: Only one episode will be "
#     "sampled.",
# )

# if __name__ == "__main__":
#     args = parser.parse_args()

#     def env_creator(args):
#         return ParallelPettingZooEnv(DepotEnv(num_agents=5))

#     env = env_creator({})
#     register_env("depotenv", env_creator)

#     config = (
#         PPOConfig()
#         .environment("depotenv")
#         .resources(num_gpus=args.num_gpus)
#         .rollouts(num_rollout_workers=2)
#         .multi_agent(
#             policies=env.get_agent_ids(),
#             policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
#         )
#     )

#     if args.as_test:
#         # Only a compilation test of running waterworld / independent learning.
#         stop = {"training_iteration": 1}
#     else:
#         stop = {"episodes_total": 60000}

#     tune.Tuner(
#         "PPO",
#         run_config=air.RunConfig(
#             stop=stop,
#             checkpoint_config=air.CheckpointConfig(
#                 checkpoint_frequency=10,
#             ),
#         ),
#         param_space=config,
#     ).fit()
