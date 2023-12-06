import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseLoss(nn.Module):

    def __init__(self,margin=0.3) -> None:
        super(PairwiseLoss,self).__init__()
        self.margin=margin

    def forward(self,score1,score2,label):
        if label==0:
            loss=F.mse_loss(score1,score2)
            loss=loss.mean()
        # score1对应的图片更模糊
        elif label==-1:
            loss=torch.max(score1+self.margin-score2,torch.tensor([0.0]))
            loss=loss.mean()
        else:
            loss=torch.max(score2+self.margin-score1,torch.tensor([0.0]))
            loss=loss.mean()
        return loss