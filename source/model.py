# Represents an entire neural network
import torch
import torch.nn as nn
import numpy.random as rand

from layer import Layer

class Model(nn.Module):
    '''
    Contains components of a neural network to be evolved
    '''
    def __init__(self, layers=None):
        '''
        Constructor
        '''
        super(Model, self).__init__()
        self.layers = []
        self.graph = nn.Sequential()
        self.avg_err = -1
        self.train_time = -1

        if layers:
            # Use provided network architecture
            self.length = len(layers)
            self.layers = layers
            self.build_graph()

    def init_random(self, max_layers, max_layer_size, layer_types, act_types, input_dim, output_dim):
        '''
        Create a random structure for the network
        '''
        self.length = rand.randint(2, max_layers) # Number of layers in network
        for i in range(self.length):
            layer_output_dim = -1
            if i == 0:
                # This is the first layer in the network, enforce input dimension to
                # match data
                layer_input_dim = input_dim
            else:
                layer_input_dim = self.layers[-1].out_features

            if i == self.length-1:
                # This is the last layer in the network, enforece output dimension to
                # match data
                layer_output_dim = output_dim
            else:
                # This is an intermediate layer, initialize with random output dimension
                layer_output_dim = rand.randint(2, max_layer_size)

            # Create layer
            layer_type = rand.choice(layer_types)
            layer_act = rand.choice(act_types)
            attrs = {
                'in_features': layer_input_dim,
                'out_features': layer_output_dim,
                'activation': layer_act
            }
            new_layer = Layer(layer_type, attrs)

            # Add layer to network
            self.layers.append(new_layer)

        self.build_graph()

    def build_graph(self):
        '''
        Decompose Layer objects into a string of PyTorch layers
        '''
        i = 0
        self.graph = nn.Sequential()
        for l in self.layers:
            for elm in l.components:
                self.graph.add_module(str(i), elm)
                i += 1

    def forward(self, x):
        '''
        Forward propogation
        '''
        return self.graph.forward(x)

    def print(self):
        '''
        Print network's layers
        '''
        print(self.graph)
