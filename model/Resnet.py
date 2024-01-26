import os
import torch.nn as nn
import torch
from torchvision import models
import pandas as pd

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for k,v in self.model.named_parameters():
            v.requires_grad=False
        # self.CNN1 = nn.Sequential(
        #     nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),#6*256*256
        #     nn.Sigmoid(),
        #     nn.MaxPool2d(kernel_size=2,stride=2)#6*128*128
        # )
        # self.CNN2 = nn.Sequential(
        #     nn.Conv2d(6, 16, kernel_size=5, stride=1), #16*124*124
        #     nn.Sigmoid(),
        #     nn.MaxPool2d(kernel_size=2,stride=2) #16*62*62
        # )
        # self.FC1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(16*62*62, 120),
        #     nn.Sigmoid(),
        #     nn.Linear(120, 84),
        #     nn.Sigmoid(),
        #     nn.Linear(84, 10)
        # )
        # self.model=nn.Sequential(self.CNN1,self.CNN2,self.FC1)
        self.blur=nn.Sequential(
            # nn.Linear(1000,500),
            nn.Linear(1000,500),
            nn.ReLU(),
            # nn.Linear(500,256),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(256,128),
            # nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Tanh()
        )
        # print(self.model)

    def forward_once(self,input):

        output=self.model(input)
        # output=torch.concat((output,feature),dim=1)
        output=self.blur(output)
        return output

    # 这里的criterion使用PairwiseLoss
    def forward(self,**contents):
        epoch=contents['epoch']
        mode=contents['mode']
        criterion=contents['criterion']
        labels=contents['labels']
        delta=contents['extra']['delta']
        input1=contents['input'][0]
        input2=None
        output1=self.forward_once(input1)
        if input2 is not None:
            output2=self.forward_once(input2)
        if mode=='train':
            loss=criterion(epoch,output1,output2,labels,contents)
            return loss
        elif mode=='eval':
            score_delta=output1-output2
            loss=criterion(epoch,output1,output2,labels,contents)
            # if os.path.exists('score_delta.csv'):
            #     df=pd.read_csv('score_delta.csv')
            # else:
            #     df=pd.DataFrame(columns=['epoch','label','pred','score_delta','img1','img2','score1','score2'])
            # df_new=pd.DataFrame({'label':labels.cpu().numpy().tolist(),'score_delta':score_delta.cpu().numpy().flatten().tolist(),'img1':list(contents[0]),'img2':list(contents[1]),'score1':output1.cpu().numpy().flatten().tolist(),'score2':output2.cpu().numpy().flatten().tolist()})
            # df_new['pred']=df_new['score_delta'].apply(lambda x:1 if x>delta else (-1 if x<-delta else 0))
            # df_new['epoch']=[epoch]*len(df_new)
            # df=pd.concat([df,df_new],axis=0)
            # df.to_csv('score_delta.csv',index=False)
            
            # with open('score_delta.txt','a') as out:
            #     for label,score,img1,img2,score1,score2 in zip(labels.cpu().numpy().tolist(),score_delta.cpu().numpy().tolist(),list(contents[0]),list(contents[1]),output1.cpu().numpy().tolist(),output2.cpu().numpy().tolist()):
            #         pred=-2
            #         if score[0]>delta:
            #             pred=1
            #         elif score[0]<-delta:
            #             pred=-1
            #         else:
            #             pred=0
            #         out.write(f'label:{label}-pred:{pred}-score:{score[0]},imgpair:{img1},{img2},scorepair:score1:{score1[0]},score2:{score2[0]}\n')
            # 超参数 delta
            score_delta[torch.abs(score_delta)<delta]=0
            score_delta[score_delta>delta]=1
            score_delta[score_delta<-delta]=-1
            return (labels.cpu().numpy().tolist(),score_delta.cpu().numpy().tolist()),loss.item()
        elif mode=='predict':
            return output1
        else:
            raise ValueError('mode should be train or eval or predict')

class BlurSingleResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for k,v in self.model.named_parameters():
            v.requires_grad=False
        self.single_blur=nn.Sequential(
            # 最后直接试一下nn.Linear(1000,2)?
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,2),
            # nn.Linear(1000,2),
        )

    def forward(self,**content):
        input=content['input']
        mode=content['mode']
        criterion=content['criterion']
        labels=content['labels']
        output=self.model(input)
        output=self.single_blur(output)
        if mode=='train':
            loss=criterion(output,labels)
            return loss
        elif mode=='eval':
            loss=criterion(output,labels)
            return (labels.cpu().numpy().tolist(),output.argmax(dim=-1).cpu().numpy().tolist(),output.softmax(dim=-1)[:,1].cpu().numpy().tolist()),loss.item()
        elif mode=='predict':
            return (output.argmax(dim=-1).cpu().numpy().tolist(),output.softmax(dim=-1)[:,1].cpu().numpy().tolist())
        else:
            raise ValueError('mode should be train or eval or predict')