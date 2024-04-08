from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from util.dataset import use_mini_batch, apply_sliding_window


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hiddent2out = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq.view(self.batch_size, -1, self.input_dim))
        predict = self.hiddent2out(lstm_out)
        return predict[:, -1, :]