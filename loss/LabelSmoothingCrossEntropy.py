import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        '''
        eps: smoothing rate
        reduction: 
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        '''
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        '''
        output: model logits, shape=[batch, seq_len, dim]
        target: label squence, shape=[batch, seq_len]
        '''
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            smooth_loss = -log_preds.sum()
        else:
            smooth_loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                smooth_loss = smooth_loss.mean()

        nll_loss = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return self.eps/c * smooth_loss + (1-self.eps) * nll_loss

# usage
# criterion = LabelSmoothingCrossEntropy()
# loss = criterion(outputs, taget)