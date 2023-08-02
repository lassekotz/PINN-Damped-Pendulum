import torch.nn as nn
import numpy as np
import pandas as pd
import math as math
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from copy import copy

class Pendulum():
    def __init__(self, damping, mass, length, theta_0, theta_dot_0, g=9.82, t_0=0, t_end=1, dt=0.001):
        self.damping_coefficient = damping
        self.mass = mass
        self.g = g
        self.L = length

        """ sim params """
        self.dt = dt
        self.t_0 = t_0
        self.t_end = t_end

        """ state vector """
        self.state_0 = [0, theta_dot_0, theta_0, t_0]  # INITIAL STATE, BOUNDARY
        self.theta_list, self.t_list = self.simulate(dt, t_end)


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

        self.t_list = [x[3] for x in self.state]
        self.theta_list = [x[2] for x in self.state]
        return self.theta_list, self.t_list

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
        x = self.out_layer(x)

        return x

lambda1, lambda2 = 0.001, 0.01
def train_PINN(training_steps):
    t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)
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
        if i % 100 == 0:
            print(f"Training progress: {(100*i/training_steps):.2f} %")
        losses.append(loss)

    return losses

# TODO: INCREASE T_END
damped_pendulum = Pendulum(damping=.1, mass=2, length=.5, theta_0=2, theta_dot_0=0, t_end=1.0, dt=0.01)

damped_pendulum_long = copy(damped_pendulum)
damped_pendulum_long.t_end = damped_pendulum_long.t_end*5
damped_pendulum_long.simulate(damped_pendulum_long.dt, damped_pendulum_long.t_end)
t_physics = torch.linspace(damped_pendulum_long.t_0, damped_pendulum_long.t_end, 50).view(-1, 1).requires_grad_(True)

PINN_model = PINN(1, 1, 32)
lr = .001
optimizer = torch.optim.Adam(PINN_model.parameters(), lr=lr)

epoch_losses = train_PINN(10000)
epoch_losses = [x.detach().item() for x in epoch_losses]

theta_NN = []

n_testpoints = 500
t_test = torch.linspace(damped_pendulum_long.t_0, damped_pendulum_long.t_end, n_testpoints)
for i in range(n_testpoints):
    theta_NN.append(PINN_model(torch.unsqueeze(torch.as_tensor(t_test[i], dtype=torch.float32), dim=-1)).item())
    #theta_NN.append(PINN_model(t_test[i]))

fig, axes = plt.subplots(2, 1)
axes[0].plot(epoch_losses)
axes[1].axvspan(damped_pendulum.t_0, damped_pendulum.t_end, color='y', alpha=0.5, lw=0)
axes[1].plot(damped_pendulum_long.t_list, damped_pendulum_long.theta_list, 'g')
axes[1].plot(t_test, theta_NN, '--r')
axes[1].scatter(t_physics.detach(), torch.zeros_like(t_physics))
axes[0].set_title("Loss over epochs. Boundary- and Physics Loss (MSE). lambda_1 = " + str(lambda1) + " lambda2 = " + str(lambda2))
axes[1].set_title("Pendulum simulated over range " + str(damped_pendulum.t_0) + "<= t <= " + str(damped_pendulum.t_end))
axes[1].legend(["Numerical Solution", "Neural Network Solution"])
#axes[1].set_xlim([0, damped_pendulum.t_end/damped_pendulum.dt])
plt.show()
