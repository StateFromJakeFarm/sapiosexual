import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from evolver import Evolver

e = Evolver(pop_size=10, max_layer_size=15, max_layers=6, input_dim=2, output_dim=1, trait_weights=[1, -0.2], num_generations=5, alpha=0.005, num_epochs=1000,
    loss_function=nn.MSELoss, device_ids=[0])

train_data = []
for i in range(4):
    inp = [float(d) for d in bin(i)[2:]]
    inp = [0.0 for _ in range(2 - len(inp))] + inp
    out = [float(int(inp[0]) ^ int(inp[1]))]

    train_data.append([torch.tensor(inp), torch.tensor(out)])

e.evolve(train_data, train_data)
