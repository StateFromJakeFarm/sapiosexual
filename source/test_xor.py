import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from evolver import Evolver

e = Evolver(pop_size=15, max_layer_size=30, max_layers=10, input_dim=4, output_dim=2, trait_weights=[1, 0], num_generations=5, alpha=0.005, num_epochs=1000,
    loss_function=nn.MSELoss, device_ids=[0], act_types=[nn.ReLU])

train_data = []
for i in range(16):
    inp = [float(d) for d in bin(i)[2:]]
    inp = [0.0 for _ in range(4 - len(inp))] + inp
    out = [float(d) for d in bin( (i >> 2) ^ (i & 3) )[2:]]
    out = [0.0 for _ in range(2 - len(out))] + out

    train_data.append([torch.tensor(inp), torch.tensor(out)])

e.evolve(train_data, train_data)
