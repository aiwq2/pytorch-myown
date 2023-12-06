import torch.nn as nn

def init_model(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
