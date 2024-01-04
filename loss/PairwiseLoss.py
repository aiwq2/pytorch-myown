import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseLoss(nn.Module):

    def __init__(self,margin=0.3) -> None:
        super(PairwiseLoss,self).__init__()
        self.margin=margin
        self.sorted_imgs=['208_AA1h0OE8_blur.jpg','161_AA1h2IAq.jpg','144_AA1h1ZG4.jpg','137_AA1h1ZF8.jpg']
        self.sorted_scores=torch.tensor([0.2,0.4,0.6,0.8],device=0)

    def forward(self,epoch,score1,score2,label,contents):

        mask_0=label==0
        mask_1=label==1
        mask_2=label==-1
        loss=torch.tensor([0.0],device=0)
        # 加上排序图片对各自分数的拟合
        # for index,sort_img in enumerate(self.sorted_imgs):
        #     img_left=[]
        #     for img in contents[0]:
        #         img_left.append(img==sort_img)
        #     img_right=[]
        #     for img in contents[1]:
        #         img_right.append(img==sort_img)
        #     img_left=torch.tensor(img_left,device=0)
        #     img_right=torch.tensor(img_right,device=0)
        #     if torch.numel(score1[img_left])>0:
        #         loss+=10.0*F.mse_loss(score1[img_left],self.sorted_scores[index].repeat(len(score1[img_left])).reshape(-1,1))
        #     if torch.numel(score2[img_right])>0:
        #         loss+=10.0*F.mse_loss(score2[img_right],self.sorted_scores[index].repeat(len(score1[img_right])).reshape(-1,1))

        # 使用循环的方法
        # for index,lb in enumerate(label):
        #     if lb==0:
        #         loss+=F.mse_loss(score1[index],score2[index])
        #     elif lb==1:
        #         if score1[index]-score2[index]<0:
        #             loss+=F.mse_loss(score1[index],score2[index])
        #         # loss+=F.relu(score2[index]-score1[index]+self.margin)
        #     else:
        #         if score1[index]-score2[index]>0:
        #             loss+=F.mse_loss(score1[index],score2[index])
        #         # loss+=F.relu(score1[index]-score2[index]+self.margin)
        # loss=100*loss/len(label)

        if any(mask_0):
            loss+=10*torch.mean(F.mse_loss(score1[mask_0],score2[mask_0]))
        # print('loss3:',10.0*torch.mean(F.mse_loss(score1[mask_0],score2[mask_0])))
        # left_1,right_1=score1[mask_1],score2[mask_1]
        # gt_1=torch.lt(left_1,right_1)
        # loss+=100*torch.mean(F.mse_loss(left_1[gt_1],right_1[gt_1]))
        # left_2,right_2=score1[mask_2],score2[mask_2]
        # gt_2=torch.gt(left_2,right_2)
        # loss+=100*torch.mean(F.mse_loss(left_2[gt_2],right_2[gt_2]))
        if any(mask_1):
            loss+=torch.mean(F.relu(score2[mask_1]-score1[mask_1]+self.margin))
        if any(mask_2):
            loss+=torch.mean(F.relu(score1[mask_2]-score2[mask_2]+self.margin))
        return loss