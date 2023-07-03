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
    def __init__(self, damping, mass, length, theta_0, theta_dot_0, g=9.82):
        self.damping_coefficient = damping
        self.mass = mass
        self.g = g
        self.L = length
        self.theta_0 = theta_0
        self.thetaDot_0 = theta_dot_0
        self.theta_list, self.t_list = self.simulate(0.001, 1)
        self.dataset = PendulumDataset(self.t_list, self.theta_list)

    def simulate(self, dt, t_end):
        theta = self.theta_0
        omega = self.thetaDot_0
        self.theta_list, self.t_list = [], []
        t = 0

        for i in range(round(t_end/dt)):
            self.theta_list.append(theta)
            self.t_list.append(t)
            omega = omega - (self.g/self.L)*math.sin(theta)*dt
            theta_dot = omega
            theta = theta + theta_dot*dt
            t = t + dt

        return self.theta_list, self.t_list


class PINN(nn.Module):
    def __init__(self, in_size, out_size, h_size):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(in_size, h_size)
        self.hidden = nn.Linear(h_size, h_size)
        self.out_layer = nn.Linear(h_size, out_size)
        self.tanh = nn.Tanh()

    def forward(self, in_data):
        x = self.input_layer(in_data)
        x = self.tanh(x)
        x = self.hidden(x)
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


def train_full(epochs):
    epoch_loss = []
    for i in range(epochs):
        epoch_loss.append(train_epoch(i))

    return epoch_loss

def train_epoch(epoch):
    cum_loss = 0
    c = 0
    for (x, y) in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        x = torch.unsqueeze(x, -1).float()
        y = y.float()
        PINN_model.train()
        preds = PINN_model.forward(x.to(device))
        loss = loss_fn(preds, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        c += 1

    return cum_loss/c

undamped_pendulum = Pendulum(damping=.1, mass=2, length=.5, theta_0=1, theta_dot_0=0)
train_loader = DataLoader(undamped_pendulum.dataset, 1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PINN_model = PINN(1, 1, 32)
loss_fn = nn.MSELoss()
lr = .001
optimizer = torch.optim.Adam(PINN_model.parameters(), lr=lr)

epoch_losses = train_full(10)

theta_NN = []
t_end = 1
dt = 0.001
undamped_pendulum.simulate(dt, t_end)
for i in range(int(t_end/dt)):
    theta_NN.append(PINN_model.forward(torch.unsqueeze(torch.as_tensor(undamped_pendulum.t_list[i], dtype=torch.float32), dim=-1)).item())

fig, axes = plt.subplots(1, 2)
sns.lineplot(data=epoch_losses, ax=axes[0])
sns.lineplot(data=undamped_pendulum.theta_list, ax=axes[1])
sns.lineplot(data=theta_NN, ax=axes[1])
axes[0].set_title("Loss over epochs")
axes[1].set_title("Pendulum")
axes[1].legend(["Num", "NN"])
axes[1].set_xlim([0, 1000])
plt.show()