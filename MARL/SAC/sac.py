import os
import torch as T
import torch.nn.functional as F
import numpy as np
from Replay_Buffer import ReplayBufferD1, ReplayBufferD2
from networks import ActorNetwork, CriticNetwork, ValueNetwork, CollectivePolicy


class Agent:
    def __init__(
        self,
        agent,
        alpha=0.0001,
        beta=0.001,
        input_dims=[8],
        env=None,
        gamma=0.99,
        n_actions=1,
        max_size=100000,
        tau=0.005,
        layer1_size=128,
        layer2_size=128,
        layer3_size=128,
        layer4_size=128,
        batch_size=128,
        reward_scale=1,
    ):
        self.gamma = gamma
        self.tau = tau
        self.D2 = ReplayBufferD2(max_size, input_dims, n_actions)
        self.D1 = ReplayBufferD1(max_size)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.agent = agent
        self.env = env
        self.collective = CollectivePolicy(
            alpha, input_dims, self.agent, name="collective"
        )
        self.best_collective_loss = float("inf")
        self.actor = ActorNetwork(
            alpha,
            input_dims,
            agent=self.agent,
            n_actions=n_actions,
            name="actor",
        )

        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions, agent, name="critic_1"
        )

        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions, agent, name="critic_2"
        )

        self.value = ValueNetwork(beta, input_dims, agent, name="value")
        self.target_value = ValueNetwork(beta, input_dims, agent, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = observation
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def predict_load(self, electricity):
        prices = np.array(electricity)
        prices = T.from_numpy(prices).float().to(self.actor.device)
        load_prediction = self.collective.forward(prices)
        if self.env.location[self.agent] == 0:
            load_prediction = T.zeros((1,))
        return load_prediction

    def rememberD1(self, electricity, loads):
        self.D1.store_transition(electricity, loads)

    def rememberD2(self, state, action, reward, new_state, done):
        self.D2.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print(f".... Saving Agent {self.agent} Models ....")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        # self.collective.save_checkpoint()

    def save_best_collective_model(self):
        print(
            f".... Saving Best Collective Model Agent {self.agent} with loss: {self.best_collective_loss} ...."
        )
        self.collective.save_checkpoint()

    def load_models(self):
        print(".... Loading Models ....")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.collective.load_checkpoint()

    def learn(self):
        # See if we've filled at least batch size of memory

        electricity, loads = self.D1.sample_buffer(self.batch_size)
        electricity = T.tensor(electricity, dtype=T.float).to(self.collective.device)
        loads = T.tensor(loads, dtype=T.float).to(self.actor.device)
        # update collective policy
        self.collective.optmizer.zero_grad()
        predicted_loads = self.collective.forward(electricity)
        collective_loss = F.mse_loss(loads, predicted_loads)
        collective_loss.backward(retain_graph=True)
        self.collective.optmizer.step()

        if abs(collective_loss.item()) < abs(self.best_collective_loss):
            self.best_collective_loss = collective_loss.item()
            self.save_best_collective_model()

        self.D2.update_buffer(self.collective)
        self.D1.clear()
        if self.D2.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.D2.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        # returns view of tensor with one less dimension
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        critic_value = self.critic_value(state, actions)

        # Update value network
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        critic_value = self.critic_value(state, actions)

        # ADD TEMPRERATURE
        actor_loss = (log_probs * 1e-2) - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def critic_value(self, state, actions):
        q1_new_policy = self.critic_1.forward(state, actions)

        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        return critic_value

    def is_ready(self):
        if self.D2.mem_cntr >= self.batch_size:
            return True

        else:
            return False
