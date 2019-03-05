# Represents a PyTorch neural network to be optimized by the Evolver object
import torch
import torch.nn as nn
import numpy.random as rand

class Model:
    '''
    Contains components of a neural network to be evolved
    '''
    def __init__(self, max_layers, max_layer_size, layer_types, input_dim, output_dim):
        '''
        Constructor
        '''
        self.layers = nn.Sequential()
        self.acc = -1
        self.train_time = -1

        # Begin by randomly-initializing the network
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

            # Initialize layer
            layer, layer_type = self.init_layer(rand.choice(layer_types), max_layer_size,
                layer_input_dim, layer_output_dim)

            # Update input dimension for next layer
            layer_input_dim = layer.out_features

            # Add layer to our container (which represents entire network architecture)
            self.layers.add_module('{}_{}'.format(layer_type, i), layer)

    def init_layer(self, layer_type, max_layer_size, input_dim, output_dim):
        '''
        Randomly initialize layer's features
        '''
        if output_dim == -1:
            # This is not the last layer in the network
            output_dim = rand.randint(1, max_layer_size)

        if layer_type == nn.Linear:
            return nn.Linear(input_dim, output_dim), 'Linear'

    def print(self):
        '''
        Print a summary of the network's features
        '''
        for layer in self.layers:
            if type(layer) == nn.Linear:
                print('Linear: {} --> {}'.format(layer.in_features, layer.out_features))
