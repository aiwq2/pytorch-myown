import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

class BlurPair(Dataset):
    def __init__(self,mode='train') -> None:
        super().__init__()
        # 配置的文件参数
        if mode=='train':
            # self.dataset=Path(__file__).parent/'compare_result_adjust_train'
            self.dataset=Path(__file__).parent/'train_once'
        if mode=='eval':
            self.dataset=Path(__file__).parent/'evl_once'

        self.train_file_dir='editor_merge'
        self.test_file_dir=Path(__file__).parent/'url_270000'

        if mode=='predict':
            self.train_file_dir='url_270000_img_1w-2w'
        self.img_pairs=[]
        self.labels=[]
        if mode=='train' or mode=='eval':
            self.root=self.dataset
            for txt in os.listdir(self.root):
                file_path=os.path.join(self.root,txt)
                with open(file_path,'r') as rd:
                    for pairs_and_labels in rd.readlines():
                        img0,img1,label=pairs_and_labels.strip().split(',')
                        self.img_pairs.append([img0,img1])
                        self.labels.append(int(label))
        elif mode=='predict':
            self.root=self.test_file_dir
            for txt in os.listdir(self.root):
                file_path=os.path.join(self.root,txt)
                with open(file_path,'r') as rd:
                    for img_name in rd.readlines():
                        self.img_pairs.append([img_name.strip(),None])
                        # 0为占位符，随便取的值
                        self.labels.append(0)
        else:
            raise ValueError('mode should be train or predict')
        if mode=='train' or mode=='eval':
            self.preprocess=transforms.Compose([
                transforms.Resize((256,256)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocess=transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])


    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):

        img0,img1=self.img_pairs[index]
        label=self.labels[index]
        path0=Path(__file__).parent/self.train_file_dir/img0
        PIL_img0=Image.open(path0).convert('RGB')
        PIL_img0=self.preprocess(PIL_img0)
        if img1:
            path1=Path(__file__).parent/self.train_file_dir/img1
            PIL_img1=Image.open(path1).convert('RGB')
            PIL_img1=self.preprocess(PIL_img1)
            return [PIL_img0,PIL_img1],label,[img0,img1]
        else:
            # 最后都要返回img_path用于数据分析
            return [PIL_img0,img0],label,img0

    def calculate_local_clarity(self,image_path, grid_size=(5, 5)):
        # 总特征数为grid_size[0]*grid_size[1]+4
        image = cv2.imread(image_path, 0)

        height, width= image.shape
        
        M, N = grid_size
        Y, X = int(height/M), int(width/N)
        
        stirdeY=Y/3*2
        strideX=X/3*2
        height_cal=Y*M
        width_cal=X*N
        scores = []

        entire_laplacian_var=cv2.Laplacian(image,cv2.CV_64F).var()
        # 统计横排相邻单元格是否相似
        similar_adjacent_count=0
        lasty=0
        last_score=0
        count_all=0.0
        y=0
        x=0 
        for i in range(M):
            for j in range(N):
                count_all+=1.0
                region = image[y:y+Y, x:x+X]
                laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
                scores.append(laplacian_var)
                if y==lasty:
                    if math.fabs(last_score-laplacian_var)<5.0:
                        similar_adjacent_count+=1
                lasty=y
                last_score=laplacian_var
                x+=strideX
            y+=stirdeY
        scores=np.array(scores,dtype=int)
        # 注意返回的第一个值为文件夹分类的依据
        features=[]
        # features=[int(entire_laplacian_var),height+width,int(np.max(scores)),int(similar_adjacent_count/count_all*100)]
        features.extend(scores.flatten().tolist())
        print(len(features))
        return torch.tensor(features)

class BlurPairFeature(Dataset):
    def __init__(self,args) -> None:
        super().__init__()
        self.img_pairs=[]
        self.labels=[]
        self.args=args
        if args['mode']=='train':
            self.root=Path(__file__).parent/'compare_result_adjust_train'
            for txt in os.listdir(self.root):
                file_path=os.path.join(self.root,txt)
                with open(file_path,'r') as rd:
                    for pairs_and_labels in rd.readlines():
                        img0,img1,label=pairs_and_labels.split(',')
                        self.img_pairs.append([img0,img1])
                        self.labels.append(int(label))
        elif args['mode']=='predict':
            self.root=Path(__file__).parent/'test'
            for txt in os.listdir(self.root):
                file_path=os.path.join(self.root,txt)
                with open(file_path,'r') as rd:
                    for img_name in rd.readlines():
                        self.img_pairs.append([img_name.strip(),None])
                        # 0为占位符，随便取的值
                        self.labels.append(0)
        else:
            raise ValueError('mode should be train or predict')

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        label=self.labels[index]
        img0,img1=self.img_pairs[index]
        path0=Path(__file__).parent/'editor'/img0
        feature0=self.calculate_local_clarity(str(path0))
        if img1:
            path1=Path(__file__).parent/'editor'/img1
            feature1=self.calculate_local_clarity(str(path1))
            return [feature0,feature1],label,[img0,img1]
        else:
            return [feature0,img0],label,[img0,img1]

    def calculate_local_clarity(self,image_path, grid_size=(10, 10)):
        # 总特征数为grid_size[0]*grid_size[1]+4
        image = cv2.imread(image_path, 0)

        height, width= image.shape
        
        M, N = grid_size
        Y, X = int(height/M), int(width/N)
        
        stirdeY=Y//3*2
        strideX=X//3*2
        height_cal=Y*M
        width_cal=X*N
        scores = []

        entire_laplacian_var=cv2.Laplacian(image,cv2.CV_64F).var()
        # 统计横排相邻单元格是否相似
        similar_adjacent_count=0
        lasty=0
        last_score=0
        count_all=0.0
        y=0
        x=0 
        for i in range(M):
            for j in range(N):
                count_all+=1.0
                region = image[y:y+Y, x:x+X]
                laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
                scores.append(laplacian_var)
                if y==lasty:
                    if math.fabs(last_score-laplacian_var)<5.0:
                        similar_adjacent_count+=1
                lasty=y
                last_score=laplacian_var
                x+=strideX
            y+=stirdeY
            x=0
        scores=np.array(scores,dtype=int)
        # 注意返回的第一个值为文件夹分类的依据
        features=[]
        # features=[int(entire_laplacian_var),height+width,int(np.max(scores)),int(similar_adjacent_count/count_all*100)]
        features.extend(scores.flatten().tolist())
        features_tensor=torch.tensor(features,dtype=torch.float32)
        # features_tensor=F.normalize(features_tensor,dim=0)
        # print(features_tensor)
        return features_tensor

class BlurSingle(Dataset):
    # 图片标签的名字
    BLURRY='blur'
    CLEAR='clear'
    def __init__(self,mode='train') -> None:
        super().__init__()
        self.train_file_dir=Path(__file__).parent/'gptv_2cls'
        # self.test_file_dir=Path(__file__).parent/'url_100000'
        if mode=='train':
            self.train_file_dir=Path(__file__).parent/'iter9_image/train_iterate'
        if mode=='eval':
            self.train_file_dir=Path(__file__).parent/'iter9_image/val'
        # self.predict_file_dir=Path(__file__).parent/'test_blur_iter2_single_img'
        self.predict_file_dir=Path(__file__).parent/'improve_test'
        self.imgs=[]
        if mode=='train' or mode=='eval':
            self.root=self.train_file_dir
            for label in os.listdir(self.root):
                label_path=os.path.join(self.root,label)
                for img_name in os.listdir(label_path):
                    image_path=os.path.join(label_path,img_name)
                    self.imgs.append(image_path)
        elif mode=='predict':
            for img_name in os.listdir(self.predict_file_dir):
                image_path=os.path.join(self.predict_file_dir,img_name)
                self.imgs.append(image_path)
        else:
            raise ValueError('mode should be train or predict')
        
        if mode=='train' or mode=='eval':
            self.preprocess=transforms.Compose([
                transforms.Resize((256,256)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocess=transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path=self.imgs[index]
        label=1 if self.BLURRY in img_path else 0
        img=self.preprocess(Image.open(img_path).convert('RGB'))
        return img,label,img_path
