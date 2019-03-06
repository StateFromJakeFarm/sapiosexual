# Represents a PyTorch neural network to be optimized by the Evolver object
import torch
import torch.nn as nn
import numpy.random as rand

class Model(nn.Module):
    '''
    Contains components of a neural network to be evolved
    '''
    def __init__(self, layers=None):
        '''
        Constructor
        '''
        super(Model, self).__init__()
        self.layers = nn.Sequential()
        self.avg_err = -1
        self.train_time = -1

        if layers:
            # Use provided network architecture
            self.layers = layers

    def init_random(self, max_layers, max_layer_size, layer_types, input_dim, output_dim):
        '''
        Create a random structure for the network
        '''
        layer_input_dim = -1
        self.length = rand.randint(2, max_layers) # Number of layers in network
        for i in range(self.length):
            layer_output_dim = -1
            if i == 0:
                # This is the first layer in the network, enforce input dimension to
                # match data
                layer_input_dim = input_dim

            if i == self.length-1:
                # This is the last layer in the network, enforece output dimension to
                # match data
                layer_output_dim = output_dim
            else:
                # This is an intermediate layer, initialize with random output dimension
                layer_output_dim = rand.randint(1, max_layer_size)

            # Initialize layer
            layer = self.init_layer(rand.choice(layer_types), max_layer_size,
                layer_input_dim, layer_output_dim)

            # Update input dimension for next layer
            layer_input_dim = layer.out_features

            # Add layer to our container (which represents entire network architecture)
            self.layers.add_module(str(i), layer)


    def init_layer(self, layer_type, max_layer_size, input_dim, output_dim):
        '''
        Randomly initialize layer's features
        '''
        if layer_type == nn.Linear:
            return nn.Linear(input_dim, output_dim)

    def forward(self, x):
        '''
        Forward propogation
        '''
        return self.layers.forward(x)

    def print(self):
        '''
        Print a summary of the network's features
        '''
        for layer in self.layers:
            if type(layer) == nn.Linear:
                print('Linear: {} --> {}'.format(layer.in_features, layer.out_features))
