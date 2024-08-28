import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# import torch.distributions.normal as Normal
from torch.distributions import Normal
import numpy as np


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        n_actions,
        agent,
        fc1_dims=128,
        fc2_dims=128,
        fc3_dims=128,
        fc4_dims=128,
        name="critic",
        chkpt_dir="tmp3/sac",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = [input_dims[0] + 1]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.agent = str(agent)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.name + self.agent + "_sac"
        )

        # Critic evaluates value of state action pair
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self._initialize_weights()

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc3(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc4(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    # estimates value of state or set of states
    def __init__(
        self,
        beta,
        input_dims,
        agent,
        fc1_dims=128,
        fc2_dims=128,
        fc3_dims=128,
        fc4_dims=128,
        name="value",
        chkpt_dir="tmp3/sac",
    ):
        super(ValueNetwork, self).__init__()
        self.input_dims = [input_dims[0] + 1]
        self.agent = str(agent)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.name + self.agent + "_sac"
        )

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self._initialize_weights()

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        agent,
        fc1_dims=128,
        fc2_dims=128,
        fc3_dims=128,
        fc4_dims=128,
        n_actions=1,
        name="actor",
        chkpt_dir="tmp3/sac",
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = [input_dims[0] + 1]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.name = name
        self.agent = str(agent)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.name + self.agent + "_sac"
        )
        self.reparam_noise = 1e-6  # makes sure we dont take log of 0
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        # mean of distribution for policy
        self.mu = nn.Linear(self.fc4_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc4_dims, self.n_actions)  # standard deviation

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self._initialize_weights()

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = self.fc3(prob)
        prob = F.relu(prob)
        prob = self.fc4(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.mu(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)

        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class CollectivePolicy(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        agent,
        fc1_dims=128,
        fc2_dims=128,
        fc3_dims=128,
        fc4_dims=128,
        name="collective",
        chkpt_dir="tmp3/sac",
    ):
        super(CollectivePolicy, self).__init__()
        self.input_dims = [24]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims

        self.name = name
        self.agent = str(agent)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.name + self.agent + "_sac"
        )

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.x = nn.Linear(self.fc2_dims, 1)

        self.optmizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self._initialize_weights()

    def forward(self, prices):
        approx = self.fc1(prices)
        approx = F.relu(approx)
        approx = self.fc2(approx)
        approx = F.relu(approx)
        approx = self.fc3(approx)
        approx = F.relu(approx)
        approx = self.fc4(approx)
        approx = F.relu(approx)

        x = self.x(approx)

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


"""The collective-policy model is constructed using DNN, which is used to approximate the behaviors of other agents."""
