import math
import random
import numpy as np
import torch
import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import IPython
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import argparse
import time

GPU = True
device_idx = 0

if GPU:
    device = torch.device(
        "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu"
    )
else:
    device = torch.device("cpu")

print(device)

parser = argparse.ArgumentParser(
    description="Train or test neural net motor controller."
)
parser.add_argument("--train", dest="train", action="store_true", default=False)
parser.add_argument("--test", dest="test", action="store_true", default=False)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, max_cap):
        self.max_cap = max_cap
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_cap:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.max_cap)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 1)

        self.layer4.weight.data.uniform_(-init_w, init_w)
        self.layer4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(CriticNetwork, self).__init__()

        self.layer1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 1)

        self.layer4.weight.data.uniform_(-init_w, init_w)
        self.layer4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class ActorNetwork(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_range=1.0,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(ActorNetwork, self).__init__()

        self.action_range = action_range
        self.num_actions = num_actions

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = nn.Linear(num_inputs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, num_actions)
        self.mean_layer.weight.data.uniform_(-init_w, init_w)
        self.mean_layer.bias.data.uniform_(-init_w, init_w)

        self.log_std_layer = nn.Linear(hidden_dim, num_actions)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """
        Generates actions and calculates the likelihood (log probability of those actions)
        Then take a sample from a standard normal distribution
        """

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))
        action = self.action_range * action_0

        log_prob = (
            Normal(mean, std).log_prob(mean + std * z.to(device))
            - torch.log(1.0 - action_0.pow(2) + epsilon)
            - np.log(self.action_range)
        )

        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = (
            self.action_range * torch.tanh(mean).detach().cpu().numpy()[0]
            if deterministic
            else action.detach().cpu().numpy()[0]
        )

    def sample_action(
        self,
    ):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range * a.numpy()


class SAC_Trainer:
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        state_dim = None  #####
        action_dim = None  #####

        # Initialize  the actor and critic networks
        self.critic_net1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_net2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_net1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.target_critic_net2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.actor_net = ActorNetwork(
            state_dim, action_dim, hidden_dim, action_range
        ).to(device)

        self.log_alpha = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=device
        )
        print("Critic Network (1,2): ", self.critic_net1)
        print("Actor Network:, ", self.actor_net)

        """Copy the parameters from the main networks to their corresponding target networks to stabilize training """
        for target_param, param in zip(
            self.target_critic_net1.parameters(), self.critic_net1.parameters()
        ):
            target_param.data.copy_(param.data)

        for tatget_param, param in zip(
            self.target_critic_net2.parameters(), self.target_critic_net2.parameters()
        ):
            target_param.data.copy_(param.data)

        self.critic1_loss = nn.MSELoss()
        self.critic2_loss = nn.MSELoss()

        critic_lr = 3e-4
        actor_lr = 3e-4
        alpha_lr = 3e-4

        self.critic1_optimizer = optim.Adam(self.critic_net1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic_net2.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.alpa_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(
        self,
        batch_size,
        reward_scale=10.0,
        auto_entropy=True,
        target_entropy=-2,
        gamma=0.99,
        soft_tau=1e-2,
    ):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        pred_q_value1 = self.critic_net1(state, action)
        pred_q_value2 = self.critic_net2(state, action)

        new_action, log_prob, z, mean, log_std = self.actor_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.actor_net.evaluate(next_state)

        reward = (
            reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
        )

        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpa_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpa_optimizer.step()
            self.alpha = self.log_alpha.exp()

        else:
            self.aplha = 1
            alpha_loss = 0

        target_q_min = (
            torch.min(
                self.target_critic_net1(next_state, new_next_action),
                self.target_critic_net2(next_state, new_next_action),
            )
            - self.alpha * next_log_prob
        )
        target_q_value = reward + (1 - done) * gamma * target_q_min
        q_value_loss1 = self.critic1_loss(pred_q_value1, target_q_value.detach())
        q_value_loss2 = self.critic2_loss(pred_q_value2, target_q_value.detach())

        self.critic1_optimizer.zero_grad()
        q_value_loss1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        q_value_loss2.backward()
        self.critic2_optimizer.step()

        pred_new_q_value = torch.min(
            self.critic_net1(state, new_action), self.critic_net2(state, new_action)
        )
        actor_loss = (self.alpha * log_prob - pred_new_q_value).mean()
        self.actor_optimizer.zero_grad
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(
            self.target_critic_net1.parameters(), self.critic_net1.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(
            self.target_critic_net2.parameters(), self.critic_net2.parameters()
        ):
            target_param.data.copy_(
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            )

        return pred_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.critic_net1.state_dict(), path + "_critic1")
        torch.save(self.critic_net2.state_dict(), path + "_critic2")
        torch.save(self.actor_net.state_dict(), path + "_actor")

    def load_model(self, path):
        self.critic_net1.load_state_dict(torch.load(path + "_critic1"))
        self.critic_net2.load_state_dict(torch.load(path + "_critic2"))
        self.actor_net.load_state_dict(torch.load(path + "_actor"))

        self.critic_net1.eval()
        self.critic_net2.eval()
        self.actor_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig("sac.png")


buffer_size = 1e6
buffer = ReplayBuffer(buffer_size)

batch_size = 300
explore_steps = 0
update_itr = 1
action_range = 350.0
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = 512
rewards = []
model_path = "./model/sac"
max_episodes = 1000
model = SAC_Trainer(buffer, hidden_dim=hidden_dim, action_range=action_range)
env = None  #####
done = False  #####
max_steps = None  ######
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
if __name__ == "__main__":
    if args.train:
        for eps in range(max_episodes):
            state = env.reset()
            ep_reward = 0

            for step in range(max_steps):
                action = model.actor_net.get_action(state, deterministic=DETERMINISTIC)
                next_state, reward, done = env.step(action)

                buffer.push(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward

                if len(buffer) > batch_size:
                    for i in range(update_itr):
                        _ = model.update(
                            batch_size,
                            reward_scale=10.0,
                            auto_entropy=AUTO_ENTROPY,
                            target_entropy=1.0 * action_dim,
                        )

            if done:
                break
        if eps % 20 == 0 and eps > 0:
            plot(rewards)
            np.save("rewards", rewards)
            model.save_model(model_path)
        print("Episode: ", eps, "| Episode Reward: ", ep_reward)
        rewards.append(ep_reward)
    model.save_model(model_path)
