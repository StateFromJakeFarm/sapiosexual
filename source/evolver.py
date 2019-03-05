# Simulates evolution of neural networks
import time
import torch
import torch.nn as nn
import numpy.random as rand

from model import Model

class Evolver:
    def __init__(self, max_layers, max_layer_size, layer_types, input_dim, output_dim,
        pop_size, num_generations, loss_function, optimizer, desirable_traits, num_epochs,
        alpha):
        '''
        Constructor
        '''
        self.max_layers = max_layers
        self.max_layer_size = max_layer_size
        self.layer_types = layer_types
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.loss_function = loss_function()
        self.optimizer = optimizer
        self.desirable_traits = desirable_traits
        self.num_epochs = num_epochs
        self.alpha = alpha

        self.pop = []

    def init_pop(self):
        '''
        Create a starting population of random individuals
        '''
        params = (self.max_layers, self.max_layer_size, self.layer_types, self.input_dim, self.output_dim)
        self.pop = [Model(*params) for _ in range(self.pop_size)]

    def train(self, member, train_set):
        '''
        Train a model and return time to train
        '''
        optimizer = self.optimizer(member.parameters(), lr=self.alpha)
        start = time.time()

        for epoch in range(self.num_epochs):
            for sample in train_set:
                # Run data through network
                output = member(sample[0])

                # Calculate error
                loss = self.loss_function(output, sample[1])

                # Backpropagate errors
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return time.time() - start

    def test(self, member, test_set):
        '''
        Test a model and return error against test set
        '''
        err = 0
        for sample in test_set:
            output = member(sample[0])
            err += self.loss_function(output, sample[1])

        return err

    def evolve(self, train_set, test_set):
        '''
        Simulate evolution of neural networks
        '''
        # Initialize the beginning population
        self.init_pop()

        # Simulate each generation
        for gen in range(self.num_generations):
            # Train each member of the population
            for i, member in enumerate(self.pop):
                member = member.float()
                train_time = self.train(member, train_set)
                acc = self.test(member, test_set)

                print('{}/{}: acc = {} train time = {}'.format(
                    i, self.pop_size, acc, train_time))
