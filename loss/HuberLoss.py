# 跟Smooth L1 Loss比较接近，也是平方损失和绝对损失的综合，只不过多了delta这一超参数
delta=1.0  # 先定义超参数

def huber_loss(x,y):
    assert len(x)==len(y)
    loss=0
    for i_x,i_y in zip(x,y):
        tmp = abs(i_y-i_x)
        if tmp<=delta:
            loss+=0.5*(tmp**2)
        else:
            loss+=tmp*delta-0.5*delta**2
    return loss