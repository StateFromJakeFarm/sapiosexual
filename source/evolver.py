# Simulates evolution of neural networks
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as rand

from model import Model

class Evolver:
    def __init__(self, max_layers=10, max_layer_size=10, layer_types=[nn.Linear],
        input_dim=10, output_dim=10, pop_size=10, num_generations=10, loss_function=nn.L1Loss,
        optimizer=optim.Rprop, trait_weights=[1, -1], num_epochs=100, mutation_pct=0.2,
        alpha=0.001):
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
        self.trait_weights = trait_weights
        self.num_epochs = num_epochs
        self.mutation_pct = mutation_pct
        self.alpha = alpha

        self.pop = []

    def init_pop(self):
        '''
        Create a starting population of random individuals
        '''
        params = (self.max_layers, self.max_layer_size, self.layer_types, self.input_dim, self.output_dim)

        self.pop = [Model() for _ in range(self.pop_size)]
        for member in self.pop:
            # Each member begins as a random network
            member = member.float()
            member.init_random(*params)

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

    def test(self, member, test_set, print_outputs=False):
        '''
        Test a model and return average error against test set
        '''
        err = 0.0
        for sample in test_set:
            output = member(sample[0])
            err += self.loss_function(output, sample[1])

            if print_outputs:
                print(sample[0], output)

        return err/len(test_set)

    def rank_members(self):
        '''
        Rank members based on desirable trait weights
        '''
        self.pop = sorted(self.pop, key=lambda m: m.avg_err*self.trait_weights[0] + m.train_time*self.trait_weights[1])

    def gen_mutation(self):
        layer_type = rand.choice(self.layer_types)
        if layer_type == nn.Linear:
            input_dim = rand.randint(1, self.max_layer_size)
            output_dim = rand.randint(1, self.max_layer_size)
            return nn.Linear(input_dim, output_dim)

    def breed(self, parents):
        '''
        Mix features of parents and add mutations to produce next generation
        '''
        self.members = []
        for i, p1 in enumerate(parents[:2:]):
            if i*2+1 >= len(parents):
                p2 = parents[0]
            else:
                p2 = parents[i*2+1]

            # Must include first and last layers
            p1_ind = rand.randint(1, p1.length-1)
            p2_ind = rand.randint(1, p2.length-1)

            layers = nn.Sequential()
            for i in range(p1_ind):
                if i > 0 and rand.ranf() < self.mutation_pct:
                    mutated_layer = self.gen_mutation()
                    layers.add_module(str(i), mutated_layer)
                else:
                    layers.add_module(str(i), p1.layers[i])
            for i in range(p2.length - p2_ind):
                if i < p2.length-p2_ind-1 and rand.ranf() < self.mutation_pct:
                    mutated_layer = self.gen_mutation()
                    layers.add_module(str(i), mutated_layer)
                else:
                    layers.add_module(str(p2_ind+i), p2.layers[p2_ind+i])

            self.members.append(Model(layers=layers))

    def print_stats(self):
        '''
        Print general information about the population
        '''
        avg_len = np.mean([m.length for m in self.pop])
        avg_layer_size = np.mean([np.mean([l.in_features + l.out_features for l in m.layers]) for m in self.pop])
        avg_avg_err = np.mean([float(m.avg_err) for m in self.pop])
        avg_train_time = np.mean([float(m.train_time) for m in self.pop])

        print('  avg_len = {}\n  avg_layer_size = {}\n  avg_avg_err = {}\n  avg_train_time = {}'.format(
            avg_len, avg_layer_size, avg_avg_err, avg_train_time))

    def evolve(self, train_set, test_set):
        '''
        Simulate evolution of neural networks
        '''
        # Initialize the beginning population
        self.init_pop()

        # Simulate each generation
        for gen in range(self.num_generations):
            print('Generation {}'.format(gen+1))
            # Train each member of the population
            for i, member in enumerate(self.pop):
                member.train_time = self.train(member, train_set)
                member.avg_err = self.test(member, test_set)

                print('  {}/{}: avg_err = {} train time = {}'.format(
                    i+1, self.pop_size, member.avg_err, member.train_time))

            self.print_stats()

            # Discover the most "fit" members
            self.rank_members()

            # Top members mate
            parents = self.pop[:self.pop_size//2]
            self.breed(parents)
            print()

        # Print out structure of top member and test it
        top_member = self.pop[0]
        top_member.print()
        self.test(top_member, test_set, print_outputs=True)
