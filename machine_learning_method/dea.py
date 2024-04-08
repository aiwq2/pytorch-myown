import os
import math
# 比较深度学习和机器学习的结果的差异
file_ml='predict_result.txt'
file_dl='resnet_predict_result.txt'

ml_dict={}
dl_dict={}
with open(file_ml,'r') as f:
    with open(file_dl,'r') as f2:
        dir_name=os.path.dirname(f.readline().strip().split(',')[0])
        print(f'dir_name:{dir_name}')
        lines_ml=f.readlines()
        image_name_list=[]
        for line_ml in lines_ml:
            imgae_name,label,pred=line_ml.strip().split(',')
            ml_dict[imgae_name]=pred
            image_name_list.append(imgae_name)
            
        lines_dl=f2.readlines()
        for line_dl in lines_dl:
            imgae_name,label,pred=line_dl.strip().split(',')
            real_name=dir_name+'/'+os.path.basename(imgae_name)
            dl_dict[real_name]=pred
with open('compare.txt','w') as out:
    for image_name in image_name_list:
        ml_pred=float(ml_dict[image_name])
        dl_pred=float(dl_dict[image_name])
        if ml_pred>=0.4 and ml_pred<=0.6:
            out.write(f'{image_name},{ml_pred}\n')
        elif (ml_pred<=0.2 and dl_pred>=0.5) or (ml_pred>=0.8 and dl_pred<=0.5) or (ml_pred>=0.5 and dl_pred<=0.2) or (ml_pred<=0.5 and dl_pred>=0.8):
            out.write(f'{image_name},{ml_pred},{dl_pred}\n')
