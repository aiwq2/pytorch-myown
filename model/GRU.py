from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from util.dataset import use_mini_batch, apply_sliding_window


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size=1):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hiddent2out = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq):
        gru_out, _ = self.gru(seq.view(self.batch_size, -1, self.input_dim))
        predict = self.hiddent2out(gru_out)
        return predict[:, -1, :]