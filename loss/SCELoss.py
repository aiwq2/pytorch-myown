import torch.nn as nn
import torch.nn.functional as F
import torch

# 针对有噪声样本对CELoss进行优化，感觉也相当于标签平滑smooth labeling
class SCELoss(nn.Module):
    def __init__(self, num_classes=36, a=1, b=0.1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-4, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss