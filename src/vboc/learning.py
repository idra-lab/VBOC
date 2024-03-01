import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network with 3 layers (1 hidden). """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


class RegressionNN:
    """ Class that compute training and test of a neural network. """
    def __init__(self, model, loss_fn, optimizer, beta, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = beta
        self.batch_size = batch_size

    def training(self, x_train, y_train):
        """ Training of the neural network. """
        t = 1
        n = len(x_train)
        val = np.amax(y_train)
        b = int(n * 100 / self.batch_size)          # number of iterations for 100 epochs
        max_iter = b * 10
        evolution = []
        self.model.train()
        while val > 1e-3 and t < max_iter:
            indexes = random.sample(range(n), self.batch_size)

            x_tensor = torch.Tensor(x_train[indexes]).to(self.device)
            y_tensor = torch.Tensor(y_train[indexes]).to(self.device)

            # Forward pass: compute predicted y by passing x to the model
            y_pred = self.model(x_tensor)

            # Compute the loss
            loss = self.loss_fn(y_pred, y_tensor)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            val = self.beta * val + (1 - self.beta) * loss.item()
            t += 1
            if t % b == 0:
                print(f'Iteration {t}, loss = {val}')
                evolution.append(val)
        return evolution

    def testing(self, x_test, y_test):
        """ Compute the RMSE wrt to training or test data. """
        loader = DataLoader(torch.Tensor(x_test).to(self.device), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        y_pred = np.empty((len(x_test), 1))
        with torch.no_grad():
            for i, x in enumerate(loader):
                if (i + 1) * self.batch_size > len(x_test):
                    y_pred[i * self.batch_size:] = self.model(x).cpu().numpy()
                else:
                    y_pred[i * self.batch_size:(i+1) * self.batch_size] = self.model(x).cpu().numpy()
        return y_pred, np.sqrt(np.mean((y_pred - y_test)**2))


def plot_viability_kernel(nq, params, model, mean, std, grid=1e-2):
    # Create the grid
    q, v = np.meshgrid(np.arange(params.q_min, params.q_max, grid),
                       np.arange(params.dq_min, params.dq_max, grid))
    q_rav, v_rav = q.ravel(), v.ravel()
    n = len(q_rav)

    for i in range(nq):
        with torch.no_grad():
            plt.figure()

            x = np.zeros((n, nq * 2))
            x[:, :nq] = (params.q_max + params.q_min) / 2 * np.ones((n, nq))
            x[:, i] = q_rav
            x[:, nq + i] = v_rav

            # Compute velocity norm
            y = np.linalg.norm(x[:, nq:], axis=1)

            # Normalize position
            x[:, :nq] = (x[:, :nq] - mean) / std
            # Velocity direction
            x[:, nq:] /= y.reshape(len(y), 1)

            # Predict
            y_pred = model(torch.from_numpy(x.astype(np.float32))).cpu().numpy()
            out = np.array([0 if y[j] > y_pred[j] else 1 for j in range(n)])
            z = out.reshape(q.shape)
            plt.contourf(q, v, z, cmap='coolwarm', alpha=0.8)
            plt.xlim([params.q_min, params.q_max])
            plt.ylim([params.dq_min, params.dq_max])
            plt.xlabel('q_' + str(i + 1))
            plt.ylabel('dq_' + str(i + 1))
            plt.grid()
            plt.title(f"Classifier section joint {i + 1}")
