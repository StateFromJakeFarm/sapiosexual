# Simulates evolution of neural networks
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as rand

from copy import deepcopy
from model import Model
from layer import Layer

class Evolver:
    def __init__(self, max_layers=10, max_layer_size=10, layer_types=[nn.Linear],
        act_types=[nn.Sigmoid, nn.ReLU, nn.Tanh], input_dim=10, output_dim=10, pop_size=10,
        num_generations=10, loss_function=nn.L1Loss, optimizer=optim.Rprop,
        trait_weights=[1, -1], num_epochs=100, mutation_pct=0.2, parent_pct=0.25,
        alpha=0.001, device_ids=None):
        '''
        Constructor
        '''
        self.max_layers = max_layers
        self.max_layer_size = max_layer_size
        self.layer_types = layer_types
        self.act_types = act_types
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.loss_function = loss_function()
        self.optimizer = optimizer
        self.trait_weights = trait_weights
        self.num_epochs = num_epochs
        self.mutation_pct = mutation_pct
        self.parent_pct = parent_pct
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
        params = (self.max_layers, self.max_layer_size, self.layer_types, self.act_types, self.input_dim,
            self.output_dim)

        self.pop = [Model() for _ in range(self.pop_size)]
        for member in self.pop:
            # Each member begins as a random network
            member = member.float()
            member.init_random(*params)

    def train(self, member, train_set):
        '''
        Train a model and return time to train
        '''
        optimizer = self.optimizer(member.graph.parameters(), lr=self.alpha)
        start = time.time()

        for epoch in range(self.num_epochs):
            for sample in train_set:
                # Prep data for use on device
                sample[0] = sample[0].to(self.device)
                sample[1] = sample[1].to(self.device)

                # Run data through network
                output = member.graph(sample[0]).to(self.device)

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
            # Prep data for use on device
            sample[0] = sample[0].to(self.device)
            sample[1] = sample[1].to(self.device)

            output = member(sample[0])
            err += self.loss_function(output, sample[1])

            if print_outputs:
                print(sample[0], sample[1], output)

        return err/len(test_set)

    def rank_members(self):
        '''
        Rank members based on desirable trait weights
        '''
        self.pop = sorted(self.pop, key=lambda m: m.avg_err*self.trait_weights[0] + m.train_time*self.trait_weights[1])

    def crossover(self, p1, p2):
        '''
        Create child by crossing traits (layers) of two parents
        '''
        # Grab some number of layers from the start of p1's graph
        p1_num_layers = rand.randint(1, len(p1.layers))
        p1_layers = [deepcopy(l) for l in p1.layers[:p1_num_layers]]

        # Grab remaining layers from end of p2's graph
        high = min(self.max_layers - p1_num_layers, len(p2.layers))
        if high > 0:
            p2_offset = rand.randint(0, high)
        else:
            p2_offset = 1

        p2_layers = [deepcopy(l) for l in p2.layers[p2_offset:]]

        # Couple two pieces together
        p2_layers[0].in_features = p1_layers[-1].out_features
        p2_layers[0].build_components()

        return Model(layers=p1_layers + p2_layers)

    def mutate(self, member):
        '''
        Apply some mutation to member of population
        '''
        def replace_layer(member):
            # Choose random layer to replace
            i = rand.randint(0, len(member.layers))
            old_layer = member.layers[i]

            # Generate new layer
            layer_type = rand.choice(self.layer_types)
            layer_act = rand.choice(self.act_types)
            attrs = {
                'in_features': old_layer.in_features,
                'out_features': old_layer.out_features,
                'activation': layer_act
            }

            # Stitch new layer into model
            if i < len(member.layers)-1:
                # Choose random output dimension if we didn't pick the final layer
                attrs['out_features'] = rand.randint(2, self.max_layer_size)
                member.layers[i+1].in_features = attrs['out_features']
                member.layers[i+1].build_components()

            # Construct layer
            new_layer = Layer(layer_type, attrs)
            new_layer.build_components()

            # Replace layer in graph
            member.layers[i] = new_layer
            member.build_graph()

        def add_layer(member):
            if len(member.layers) >= self.max_layers:
                # Don't add layers to models that are already at capacity
                return

            # Choose random index after which the new layer will be inserted
            i = rand.randint(-1, len(member.layers))

            # Generate new layer
            layer_type = rand.choice(self.layer_types)
            layer_act = rand.choice(self.act_types)

            # Determine input features
            if i == -1:
                # First layer's input dimension must match data
                in_features = member.layers[0].in_features
            else:
                # Input dimension matches that of previous layer
                in_features = member.layers[i].out_features

            # Determine output features
            if i < len(member.layers)-1:
                # Change subsequent layer's input dimension to match that of new layer
                out_features = rand.randint(2, self.max_layer_size)
                member.layers[i+1].in_features = out_features
                member.layers[i+1].build_components()
            else:
                # Final layer must have same output dimension as data
                out_features = member.layers[-1].out_features

            attrs = {
                'in_features': in_features,
                'out_features': out_features,
                'activation': layer_act
            }

            # Create layer
            new_layer = Layer(layer_type, attrs)
            new_layer.build_components()

            # Insert new layer into graph
            member.layers.insert(i+1, new_layer)
            member.build_graph()

        def remove_layer(member):
            # Select layer to remove at random
            i = rand.randint(0, len(member.layers))

            # Stitch remaining surrounding layers together
            if i == 0:
                member.layers[1].in_features = member.layers[0].in_features
                member.layers[1].build_components()
            elif i == len(member.layers)-1:
                member.layers[i-1].out_features = member.layers[i].out_features
                member.layers[i-1].build_components()
            else:
                member.layers[i+1].in_features = member.layers[i-1].out_features
                member.layers[i+1].build_components()

            # Remove layer from graph
            del member.layers[i]
            member.build_graph()

        return {
            0: replace_layer,
            1: add_layer,
            2: remove_layer
        }[rand.randint(0, 3)](member)

    def breed(self, parents):
        '''
        Mix features of parents and add mutations to produce next generation
        '''
        self.pop = []
        i = 0
        while len(self.pop) < self.pop_size:
            p1 = parents[i]
            if i+1 >= len(parents):
                # Recycle best member if there aren't enough "top" members for
                # exclusive pairings
                p2 = parents[0]
            else:
                p2 = parents[i+1]

            # Create two children with p1 -> p2
            self.pop.append(self.crossover(p1, p2))
            self.pop.append(self.crossover(p1, p2))

            # Create two children with p2 -> p1
            self.pop.append(self.crossover(p2, p1))
            self.pop.append(self.crossover(p2, p1))

            i += 1
            if i == len(parents):
                i = 0

        # Randomly choose members of population to mutate
        for member in rand.choice(a=self.pop, size=int(self.pop_size*self.mutation_pct)):
            self.mutate(member)

        # Cut out overflow (lazy)
        self.pop = self.pop[:self.pop_size]

    def print_stats(self):
        '''
        Print general information about the population
        '''
        avg_len = np.mean([len(m.layers) for m in self.pop])
        avg_layer_size = np.mean([np.mean([l.in_features + l.out_features for l in m.layers[::2]]) for m in self.pop])
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

            if gen < self.num_generations - 1:
                # Top members breed to produce next generation
                parents = self.pop[:int(self.pop_size*self.parent_pct)]
                self.breed(parents)

                # Some portion of new population undergoes mutation to
                # maintain randomness to more fully explore the search
                # space of architectures even as certain traits become
                # more popular in the population

            print()

        # Print out structure of top member and test it
        top_member = self.pop[0]
        top_member.print()
        self.test(top_member, test_set, print_outputs=True)
