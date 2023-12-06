import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BlurPair(Dataset):
    def __init__(self,root='compare_result') -> None:
        super().__init__()
        self.img_pairs=[]
        self.labels=[]
        for txt in os.listdir(root):
            file_path=os.path.join(root,txt)
            with open(file_path,'r') as rd:
                for pairs_and_labels in rd.readlines():
                    img0,img1,label=pairs_and_labels
                    self.img_pairs.append([img0,img1])
                    self.labels.append(label)
        self.preprocess=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        img0,img1=self.img_pairs[index]
        label=self.img_pairs[index]
        PIL_img0=Image.open(img0)
        PIL_img0=self.preprocess(PIL_img0)
        PIL_img1=Image.open(img1)
        PIL_img1=self.preprocess(PIL_img1)
        return PIL_img0,PIL_img1,label

        