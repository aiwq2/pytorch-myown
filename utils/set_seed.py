import random
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)# 为cpu设置随机数种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU，为所有GPU设置随机种子


    # 可以生成随时间变化的种子，而后将其保存下来
    # seed = int(time.time() * 256)

    # 如果不需要强一致性可以不使用以下参数，避免降低训练速度
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现