import torch.nn as nn

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


    
    