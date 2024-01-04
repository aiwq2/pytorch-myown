import torch.nn as nn
import torch
import os
import pandas as pd

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=nn.Sequential( # 输入图像（1，28，28）
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2), # 输出图像（16，28，28）
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2), # 输出图像（16，14，14）
        )
        self.conv2=nn.Sequential( # 输入图像（16，14，14）
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2), # 输出图像（32，14，14）
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2), # 输出图像（32，7，7）
        )
        # self.flatten=nn.Flatten()
        self.out=nn.Linear(32*7*7,10) # 输出为10个类
    def forward(self,input):
        input=self.conv1(input)
        input=self.conv2(input)
        input=input.view(input.size(0),-1)
        output=self.out(input)
        return output


class MyCNN(nn.Module):
    '''
    构建卷积神经网络,与LeNet类似
    '''

    def __init__(self,args):
        super(MyCNN, self).__init__()
        self.args=args
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),#6*28*28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2)#6*14*14
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1), #16*10*10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2) #16*5*5
        )
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def reshape_(self, x):
        return x.reshape(-1, 1, 28, 28)

    def forward(self, mode,criterion,x,labels):
        x = self.reshape_(x)
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.FC1(x)
        if mode=='train':
            loss=criterion(x,labels)
            return loss
        elif mode=='eval':
            return (
                labels.cpu().numpy().tolist(),
                x.argmax(dim=-1).cpu().numpy().tolist(),
            )
        return x

class LinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blur=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )

    def forward_once(self,input):
        output=self.blur(input)
        return output

    def forward(self,epoch,mode,criterion,labels,contents,delta,input1,input2=None):
        output1=self.forward_once(input1)
        if input2 is not None:
            output2=self.forward_once(input2)
        if mode=='train':
            loss=criterion(output1,output2,labels)
            return loss
        elif mode=='eval':
            score_delta=output1-output2
            if os.path.exists('score_delta_LinearNet.csv'):
                df=pd.read_csv('score_delta_LinearNet.csv')
            else:
                df=pd.DataFrame(columns=['epoch','label','pred','score_delta','img1','img2','score1','score2'])
            df_new=pd.DataFrame({'label':labels.cpu().numpy().tolist(),'score_delta':score_delta.cpu().numpy().flatten().tolist(),'img1':list(contents[0]),'img2':list(contents[1]),'score1':output1.cpu().numpy().flatten().tolist(),'score2':output2.cpu().numpy().flatten().tolist()})
            df_new['pred']=df_new['score_delta'].apply(lambda x:1 if x>delta else (-1 if x<-delta else 0))
            df_new['epoch']=[epoch]*len(df_new)
            df=pd.concat([df,df_new],axis=0)
            df.to_csv('score_delta_LinearNet.csv',index=False)
            # 超参数 delta
            score_delta[torch.abs(score_delta)<delta]=0
            score_delta[score_delta>delta]=1
            score_delta[score_delta<-delta]=-1
            return labels.cpu().numpy().tolist(),score_delta.cpu().numpy().tolist()
        elif mode=='predict':
            return output1
        else:
            raise ValueError('mode should be train or eval or predict')

    
    