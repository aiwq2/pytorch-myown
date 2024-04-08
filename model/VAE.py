import torch
from time import time
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.preprocessing import minmax_scale
from util.dataset import use_mini_batch, apply_sliding_window


class VAE(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, hidden_dim)
        self.fc22 = nn.Linear(128, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.fc4 = nn.Linear(128, out_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar