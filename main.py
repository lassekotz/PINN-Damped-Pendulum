import torch.nn as nn
import numpy as np
import pandas as pd
import math as math
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Pendulum():
    def __init__(self, damping, mass, length, theta_0, theta_dot_0, g=9.82, t_0=0, t_end=10, dt=0.001):
        self.damping_coefficient = damping
        self.mass = mass
        self.g = g
        self.L = length

        """ sim params """
        self.dt = dt
        self.t_0 = t_0
        self.t_end = t_end

        """ state vector """
        self.state_0 = [0, theta_dot_0, theta_0, t_0]
        self.theta_list, self.t_list = self.simulate(dt, t_end)
        self.dataset = PendulumDataset(self.t_list, self.theta_list)


    def simulate(self, dt, t_end):

        self.state = [self.state_0]

        for i in range(round(t_end/dt)):
            phi = self.state[i][1]
            theta = self.state[i][2]
            t = self.state[i][3]

            theta += phi * dt
            psi = - phi*self.damping_coefficient/self.mass - math.sin(theta)*self.g/self.L
            phi += psi * dt
            t = t + dt
            cur_state = [psi, phi, theta, t]
            self.state.append(cur_state)


        return [x[2] for x in self.state], [x[3] for x in self.state]

class PINN(nn.Module):
    def __init__(self, in_size, out_size, h_size):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(in_size, h_size)
        self.hidden1 = nn.Linear(h_size, h_size)
        self.hidden2 = nn.Linear(h_size, h_size)
        self.out_layer = nn.Linear(h_size, out_size)
        self.tanh = nn.Tanh()

    def forward(self, in_data):
        x = self.input_layer(in_data)
        x = self.tanh(x)
        x = self.hidden1(x)
        x = self.tanh(x)
        x = self.hidden2(x)
        x = self.tanh(x)
        x = self.out_layer(x)

        return x

class PendulumDataset:
    def __init__(self, inputs, labels):
        self._pairs = self._collect_pairs(inputs, labels)

    def __getitem__(self, index):
        input_data, label = self._pairs[index]

        return input_data, label

    def __len__(self):
        """Total amount of samples in dataset"""
        return len(self._pairs)

    def _collect_pairs(self, inputs, labels):
        if len(inputs) != len(labels):
            raise ValueError("inputs and labels are of different sizes")
        else:
            datalist = []
            for i in range(len(inputs)):
                datalist.append([float(inputs[i]), float(labels[i])])

            return datalist

def train_full(epochs): # "STANDARD" TRAINING ALGORITHM
    epoch_loss = []
    for i in range(epochs):
        epoch_loss.append(train_epoch(i))

    return epoch_loss

def train_epoch(epoch): # "STANDARD" EPOCH TRAINING ALGORITHM
    cum_loss = 0
    c = 0
    for (x, y) in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        optimizer.zero_grad()
        x = torch.unsqueeze(x, -1).float()
        y = y.float()
        PINN_model.train()
        preds = PINN_model.forward(x.to(device))
        loss = loss_fn(preds, y.to(device))


        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        c += 1

    return cum_loss/c

def train_PINN(training_steps):
    lambda1, lambda2 = 0.001, 0.05
    t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)
    t_physics = torch.linspace(damped_pendulum.t_list[0], damped_pendulum.t_list[-1], 300).view(-1, 1).requires_grad_(True)

    losses = []
    for i in range(training_steps):
        optimizer.zero_grad()

        # Boundary losses:
        u = PINN_model(t_boundary)
        loss1 = (torch.squeeze(u)-damped_pendulum.state_0[2])**2
        dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
        loss2 = (torch.squeeze(dudt) - damped_pendulum.state_0[1])**2

        # Physics losses:
        u = PINN_model(t_physics)
        dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
        loss3 = torch.mean((d2udt2 + dudt*damped_pendulum.damping_coefficient/damped_pendulum.mass + torch.sin(u)*damped_pendulum.g/damped_pendulum.L)**2)

        loss = loss1 + lambda1*loss2 + lambda2*loss3
        loss.backward()
        optimizer.step()

        print(100*i/training_steps)
        losses.append(loss)

    return losses

damped_pendulum = Pendulum(damping=.1, mass=2, length=1, theta_0=1, theta_dot_0=0)
# train_loader = DataLoader(damped_pendulum.dataset, 1, shuffle=False)

PINN_model = PINN(1, 1, 32)
# loss_fn = nn.MSELoss()
lr = .001
optimizer = torch.optim.Adam(PINN_model.parameters(), lr=lr)

epoch_losses = train_PINN(10000)
epoch_losses = [x.detach().item() for x in epoch_losses]

theta_NN = []
damped_pendulum.simulate(damped_pendulum.dt, damped_pendulum.t_end)
for i in range(int(damped_pendulum.t_end/damped_pendulum.dt)):
    theta_NN.append(PINN_model.forward(torch.unsqueeze(torch.as_tensor(damped_pendulum.t_list[i], dtype=torch.float32), dim=-1)).item())

fig, axes = plt.subplots(1, 2)
sns.lineplot(data=epoch_losses, ax=axes[0])
sns.lineplot(data=damped_pendulum.theta_list, ax=axes[1])
sns.lineplot(data=theta_NN, ax=axes[1])
axes[0].set_title("Loss over epochs")
axes[1].set_title("Pendulum")
axes[1].legend(["Num", "NN"])
axes[1].set_xlim([0, damped_pendulum.t_end/damped_pendulum.dt])
plt.show()
