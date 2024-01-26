import os
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import shutil
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
import random
import pickle

def split_dataset():
    ori_normal_text='cnstd_analyze_img/Percentage of text/labels/normal_text'
    ori_much_text='cnstd_analyze_img/Percentage of text/labels/too_much_text'
    dst_normal_train_text=r'D:\workspace\python获取\imgs_20230921\cnstd_analyze_img\Percentage of text\labels\train\normal_text'
    dst_toomuach_train_text=r'D:\workspace\python获取\imgs_20230921\cnstd_analyze_img\Percentage of text\labels\train\too_much_text'
    dst_normal_test_text=r'D:\workspace\python获取\imgs_20230921\cnstd_analyze_img\Percentage of text\labels\test\normal_text'
    dst_toomuch_test_text=r'D:\workspace\python获取\imgs_20230921\cnstd_analyze_img\Percentage of text\labels\test\too_much_text'
    # image_files=[]
    for split in [ori_normal_text,ori_much_text]:
        image_files=[]
        for img in os.listdir(split):
            image_files.append(os.path.join(split,img))
        print(len(image_files))
        random.seed(42)
        random.shuffle(image_files)
        train_num=int(0.8*len(image_files))
        train_image_files=image_files[:train_num]
        test_image_files=image_files[train_num:]
        if split==ori_normal_text:
            for img in train_image_files:
                shutil.copy(img,os.path.join(dst_normal_train_text,os.path.basename(img)))
            for img in test_image_files:
                shutil.copy(img,os.path.join(dst_normal_test_text,os.path.basename(img)))
        if split==ori_much_text:
            for img in train_image_files:
                shutil.copy(img,os.path.join(dst_toomuach_train_text,os.path.basename(img)))
            for img in test_image_files:
                shutil.copy(img,os.path.join(dst_toomuch_test_text,os.path.basename(img)))