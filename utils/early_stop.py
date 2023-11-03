from typing import Any
import numpy as np
import torch

# patience：自上次模型在验证集上损失降低之后等待的时间，此处设置为7
# verbose：当为False时，运行的时候将不显示详细信息
# counter：计数器，当其值超过patience时候，使用early stopping
# best_score：记录模型评估的最好分数
# early_step：决定模型要不要early stop，为True则停
# val_loss_min：模型评估损失函数的最小值，默认为正无穷(np.Inf)
# delta：表示模型损失函数改进的最小值，当超过这个值时候表示模型有所改进

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    
    def __call__(self,val_loss,model,path):
        score=-val_loss
        if not self.best_score:
            self.best_score=score
            self.save_checkpoint(val_loss,model,path)
        elif score<self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping Counter:{self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop=True
        else:
            self.best_score=score
            self.save_checkpoint(val_loss,model,path)
            self.counter=0


    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
