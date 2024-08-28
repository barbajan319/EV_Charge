import numpy as np


class ReplayBufferD1:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.input_shape = input_shape

        self.electricity_prices_memory = np.zeros((self.mem_size, *self.input_shape))
        self.load_consumption_memory = np.zeros((self.mem_size, *input_shape))

    def store_transition(self, electricity, loads):
        index = self.mem_cntr % self.mem_size
        self.electricity_prices_memory[index] = electricity
        self.load_consumption_memory[index] = loads

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        electricity_prices = self.electricity_prices_memory[batch]
        load_consumption = self.load_consumption_memory[batch]
        return electricity_prices, load_consumption

    def clear(self):
        self.mem_cntr = 0
        self.electricity_prices_memory.fill(0)
        self.load_consumption_memory.fill(0)
