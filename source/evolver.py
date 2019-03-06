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
        alpha=0.001, device_ids=None):
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
        self.device_ids = device_ids

        self.pop = []

        # Use GPU if available
        self.device = torch.device('cpu')
        self.use_cuda = self.device_ids and torch.cuda.device_count() > 1
        if self.use_cuda:
            self.device = torch.device('cuda:0')

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
                if self.use_cuda:
                    # Prep data for use on GPU
                    sample[0] = sample[0].to(self.device)
                    sample[1] = sample[1].to(self.device)

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
            if self.use_cuda:
                # Prep data for use on GPU
                sample[0] = sample[0].to(self.device)
                sample[1] = sample[1].to(self.device)

            output = member(sample[0])
            err += self.loss_function(output, sample[1])

            if print_outputs:
                print(sample[1], output)

        return err/len(test_set)

    def rank_members(self):
        '''
        Rank members based on desirable trait weights
        '''
        self.pop = sorted(self.pop, key=lambda m: m.avg_err*self.trait_weights[0] + m.train_time*self.trait_weights[1])

    def gen_mutation(self):
        layer_type = rand.choice(self.layer_types)
        if layer_type == nn.Linear:
            input_dim = rand.randint(2, self.max_layer_size)
            output_dim = rand.randint(2, self.max_layer_size)
            return nn.Linear(input_dim, output_dim)

    def crossover(self, p1, p2):
        '''
        Create child by crossing traits (layers) of two parents
        '''
        # Must include first and last layers
        p1_ind = 1
        if p1.length > 2:
            p1_ind = rand.randint(1, p1.length-1)
        p2_ind = 1
        if p2.length > 2:
            p2_ind = rand.randint(1, p2.length-1)

        # Splice layers
        layers = nn.Sequential()
        for i in range(p1_ind+1):
            layers.add_module(str(i), p1.layers[i])

        for i in range(p2.length - p2_ind):
            if i == 0:
                layer = nn.Linear(p1.layers[p1_ind-1].out_features, p2.layers[p2_ind+i].out_features)
            else:
                layer = p2.layers[p2_ind+i]

            layers.add_module(str(p1_ind+i), layer)

        return Model(layers=layers)

    def breed(self, parents):
        '''
        Mix features of parents and add mutations to produce next generation
        '''
        self.pop = []
        for i, p1 in enumerate(parents):
            if i+1 >= len(parents):
                # Recycle best member if there aren't enough "top" members for
                # exclusive pairings
                p2 = parents[1]
            else:
                p2 = parents[i+1]

            # Create two children with p1 -> p2
            self.pop.append(self.crossover(p1, p2))
            self.pop.append(self.crossover(p1, p2))

            # Create two children with p2 -> p1
            self.pop.append(self.crossover(p2, p1))
            self.pop.append(self.crossover(p2, p1))

    def print_stats(self):
        '''
        Print general information about the population
        '''
        avg_len = np.mean([m.length for m in self.pop])
        avg_layer_size = np.mean([np.mean([l.in_features + l.out_features for l in m.layers]) for m in self.pop])
        lowest_avg_err = min([float(m.avg_err) for m in self.pop])
        avg_train_time = np.mean([float(m.train_time) for m in self.pop])

        print('  avg_len = {}\n  avg_layer_size = {}\n  lowest_avg_err = {}\n  avg_train_time = {}'.format(
            avg_len, avg_layer_size, lowest_avg_err, avg_train_time))

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
                # Use GPU if available
                member.to(self.device)
                if self.use_cuda:
                    member = torch.nn.DataParallel(member, device_ids=self.device_ids)

                member.train_time = self.train(member, train_set)
                member.avg_err = self.test(member, test_set)

                print('  {}/{}: avg_err = {} train_time = {}'.format(
                    i+1, self.pop_size, member.avg_err, member.train_time))

            self.print_stats()

            # Discover the most "fit" members
            self.rank_members()

            # Top members mate
            parents = self.pop[0:self.pop_size//4]
            self.breed(parents)
            print()

        # Print out structure of top member and test it
        top_member = self.pop[0]
        top_member.print()
        self.test(top_member, test_set, print_outputs=True)
