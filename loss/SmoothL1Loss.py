# L2损失函数的导数是动态变化的，所以x增加也会使损失增加，尤其在训练早起标签和预测的差异大，会导致梯度较大，训练不稳定。

# L1损失函数的导数为常数，在模型训练后期标签和预测的差异较小时，梯度值任然较大导致损失函数在稳定值附近波动难以进一步收敛。

# Smooth L1损失函数在x较大时，梯度为常数解决了L2损失中梯度较大破坏训练参数的问题，当x较小时，梯度会动态减小解决了L1损失中难以收敛的问题。

# Smooth L1是huber loss在delta=1时的特殊情况

def Smooth_L1(x:list,y:list):
    assert len(x)==len(y)
    loss=0
    for i_x,i_y in zip(x,y):
        tmp = abs(i_y-i_x)
        if tmp<1:
            loss+=0.5*(tmp**2)
        else:
            loss+=tmp-0.5
    return loss