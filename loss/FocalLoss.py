import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss2(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=0.20):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, inputs, targets):
        ce = self.cross_entropy(inputs, targets)
        onehot_targets = torch.nn.functional.one_hot(targets, num_classes=2)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, onehot_targets.float(), reduce=False)
        pt = torch.exp(-BCE_loss)
        FL_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return self.weight * torch.mean(FL_loss) + ce  

# # 使用示例
# loss_fn = FocalLoss(alpha=0.5, gamma=2, reduction='mean')
# inputs = torch.randn(10, 5)  # 输入模型的预测结果
# targets = torch.randint(0, 5, (10,))  # 真实标签
# loss = loss_fn(inputs, targets)
# print(loss)
