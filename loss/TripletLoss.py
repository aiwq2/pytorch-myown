

# Margin for triplet loss
# MARGIN是一个超参数，用来控制正负样本之间的距离，如果距离小于MARGIN，那么损失为0，如果距离大于MARGIN，那么损失为距离差值。
MARGIN = 0.2
import torch

def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """

        anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

        # 欧式距离
        pos_dist = torch.sum(torch.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = torch.sum(torch.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + MARGIN

        loss = torch.maximum(basic_loss, 0.0)

        print ("[INFO] model - triplet_loss shape: %s" % str(loss.shape))
        return loss