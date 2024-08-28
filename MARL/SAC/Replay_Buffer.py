import numpy as np
import torch as T


class ReplayBufferD1:
    def __init__(self, max_size, input_shape=[24]):
        self.mem_size = 24
        self.input_shape = [24]
        self.mem_cntr = 0
        self.electricity_prices_memory = np.zeros((self.mem_size, *self.input_shape))
        self.load_consumption_memory = np.zeros((self.mem_size, 1))

    def store_transition(self, electricity, loads):
        index = self.mem_cntr % self.mem_size
        self.electricity_prices_memory[index] = electricity
        self.load_consumption_memory[index] = loads

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        electricity_prices = self.electricity_prices_memory
        load_consumption = self.load_consumption_memory
        return electricity_prices, load_consumption

    def clear(self):
        self.mem_cntr = 0
        self.electricity_prices_memory.fill(0)
        self.load_consumption_memory.fill(0)


class ReplayBufferD2:
    def __init__(self, max_size, input_shape, n_actions):
        self.input_shape = [input_shape[0] + 1]
        self.mem_size = max_size
        self.mem_cntr = 0
        # *unpacks elements of the variable
        self.state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.state_memory.flags.writeable = True
        self.new_state_memory.flags.writeable = True
        self.action_memory.flags.writeable = True
        self.reward_memory.flags.writeable = True
        self.terminal_memory.flags.writeable = True

    # def store_transition(self, state, action, reward, state_, done):
    #     index = self.mem_cntr % self.mem_size
    #     # set the numpy arrays of parameters
    #     self.state_memory[index] = state
    #     self.new_state_memory[index] = state_
    #     self.action_memory[index] = action
    #     self.reward_memory[index] = reward
    #     self.terminal_memory[index] = done

    #     self.mem_cntr += 1

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        # Ensure memory arrays are writable before storing transitions
        self.state_memory = np.copy(self.state_memory)
        self.new_state_memory = np.copy(self.new_state_memory)
        self.action_memory = np.copy(self.action_memory)
        self.reward_memory = np.copy(self.reward_memory)
        self.terminal_memory = np.copy(self.terminal_memory)

        # Ensure memory arrays are writable before storing transitions
        self.state_memory.flags.writeable = True
        self.new_state_memory.flags.writeable = True
        self.action_memory.flags.writeable = True
        self.reward_memory.flags.writeable = True
        self.terminal_memory.flags.writeable = True
        try:
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = done
        except ValueError as e:
            print(f"Error occurred at index {index}")
            print(f"state_memory writable: {self.state_memory.flags.writeable}")
            print(f"new_state_memory writable: {self.new_state_memory.flags.writeable}")
            print(f"action_memory writable: {self.action_memory.flags.writeable}")
            print(f"reward_memory writable: {self.reward_memory.flags.writeable}")
            print(f"terminal_memory writable: {self.terminal_memory.flags.writeable}")
            raise e

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # set the maximum number of memories in replay buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        # Select a sequence of random int between 0 and max mem
        batch = np.random.choice(max_mem, batch_size)

        # Sample
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def update_buffer(self, policy):
        max_mem = min(self.mem_cntr, self.mem_size)
        for i in range(max_mem):

            current_state = self.state_memory[i]
            current_state = T.tensor(current_state[:24], dtype=T.float)
            new_state = policy.forward(current_state)
            self.state_memory[i][-1] = new_state

            current_state_ = self.new_state_memory[i]
            current_state_ = T.tensor(current_state_[:24], dtype=T.float)
            new_state_ = policy.forward(current_state_)
            self.new_state_memory[i][-1] = new_state_
