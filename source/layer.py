# Represents a conceptual layer in a neural network
import torch
import torch.nn as nn

class Layer:
    '''
    Single layer of a neural network
    '''
    def __init__(self, layer_type, attrs):
        '''
        Constructor
        '''
        self.layer_type = layer_type
        self.layer_attrs = attrs

        # Different layers expect different attributes
        expected_attrs = {
            nn.Linear: ['in_features', 'out_features', 'activation']
        }.get(layer_type)

        if not expected_attrs:
            # Unknown layer type
            raise ValueError('{} is not a recognized layer type'.format(str(layer)))

        # Make all passed attributes into attributes of this class
        for attr_name in expected_attrs:
            attr_val = attrs.get(attr_name)
            if not attr_val:
                # Missing required attribute
                raise ValueError('{} requires attribute {}'.format(str(layer), attr_name))

            setattr(self, attr_name, attr_val)

        # Construct layer
        self.build_components()

    def build_components(self):
        '''
        Construct components based on passed attrs
        '''
        self.components = []
        if self.layer_type == nn.Linear:
            # Linear layer needs activation function coupled with it
            self.components.append(nn.Linear(self.in_features, self.out_features))
            self.components.append(self.activation())

    def print(self):
        '''
        Print attributes
        '''
        for attr in dir(self):
            if attr[:2] != '__':
                print('{}: {}'.format(attr, str(getattr(self, attr))))
