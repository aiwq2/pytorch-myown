import torch.nn as nn
import torch
from torchvision import models

class VIT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        for k,v in self.model.named_parameters():
            v.requires_grad=False
        self.blur=nn.Sequential(
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )
        # print(self.model)

    def forward_once(self,input):
        output=self.model(input)
        output=self.blur(output)
        return output

    # 这里的criterion使用PairwiseLoss
    def forward(self,mode,criterion,labels,delta,input1,input2=None):
        output1=self.forward_once(input1)
        if input2 is not None:
            output2=self.forward_once(input2)
        if mode=='train':
            loss=criterion(output1,output2,labels)
            return loss
        elif mode=='eval':
            score_delta=output1-output2
            # 超参数 delta
            score_delta[torch.abs(score_delta)<delta]=0
            score_delta[score_delta>delta]=1
            score_delta[score_delta<-delta]=-1
            return labels.cpu().numpy().tolist(),score_delta.cpu().numpy().tolist()
        elif mode=='predict':
            return output1
        else:
            raise ValueError('mode should be train or eval or predict')

# vit=VIT()
# a=torch.randn(1,3,224,224)
# b=vit(a)
# print(b.shape)