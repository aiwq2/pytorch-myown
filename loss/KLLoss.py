import numpy as np

def kl_loss(y_true:list,y_pred:list):
    """
    y_true,y_pred，分别是两个概率分布
    比如：px=[0.1,0.2,0.8]
          py=[0.3,0.3,0.4]
    """
    assert len(y_true)==len(y_pred)
    KL=0
    for y,fx in zip(y_true,y_pred):
        KL+=y*np.log(y/fx)
    return KL