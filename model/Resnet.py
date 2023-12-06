import torch.nn as nn
import torch
from torchvision import models

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=models.resnet18(pretrained=True)
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
    def forward(self,mode,criterion,input1,intpu2,labels,delta):
        output1=self.forward_once(input1)
        output2=self.forward_once(intpu2)
        if mode=='train':
            loss=criterion(output1,output2,labels)
            return loss
        elif mode=='eval':
            score_delta=output1-output2
            # 超参数
            score_delta[torch.abs(score_delta)<delta]=0
            score_delta[score_delta>delta]=1
            score_delta[score_delta<-delta]=-1
            return labels,score_delta
        return output1,output2

rnet=ResNet()
for k,v in rnet.named_buffers():
    print(f'{k}:{v.requires_grad}')
# output=rnet(torch.randn(1,3,256,412))
# print(output)