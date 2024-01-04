import torch.nn as nn
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # 设置CNN层的Sequential
        self.cnn1 = nn.Conv2d(3, 256, kernel_size=11, stride=4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.cnn2 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.cnn3 = nn.Conv2d(256, 384, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(46464, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward_once(self, x):
        # 此函数将用于处理两个图像
        # 输出用于确定相似性
        output = self.cnn1(x)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = self.cnn2(output)
        output = self.relu(output)
        output = self.maxpool2(output)
        output = self.cnn3(output)
        output = self.relu(output)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

    def forward(self, input1, input2):
        # 在此函数中，我们传入两个图像并获取两个向量
        # 然后返回这两个向量
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2