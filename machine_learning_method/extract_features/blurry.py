import os
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import shutil
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
import random
import pickle
from pathlib import Path
import os
import cv2
import numpy as np
from scipy.stats import entropy
import math

# canny算子，分数越小越模糊
def canny(image):
    '''
    低阈值用于指定可能边缘的最小梯度值。
    所有梯度值低于低阈值的像素都会被认为不是边缘，并被直接舍弃。
    高阈值用于指定可能边缘的最大梯度值。
    所有梯度值高于高阈值的像素被认为是强边缘。
    像素的梯度值介于低阈值和高阈值之间的情况被认为是中等强度的边缘。这些像素将被保留，但只有当它们与强边缘连接时才会被认为是最终的边缘像素。
    '''
    edges=cv2.Canny(image,100,200)
    return np.mean(edges)

# 拉普拉斯算子，分数越小越模糊
def variance_of_laplacian(image):
    '''
    计算图像的laplacian响应的方差值
    '''
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 快速傅里叶变换,分数越小越模糊
def fast_fourier_transform(image, size=60):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
        
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)


    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

# 通过傅里叶变换分析图像的频率内容，模糊图像通常缺乏高频信息
def fourier_transform(image):
    f=np.fft.fft2(image)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum=20*np.log(np.abs(fshift))
    return np.mean(magnitude_spectrum)

# 图像对比度,对比度低的图像可能会显得模糊。可以通过计算图像的全局对比度或者局部对比度来评估清晰度
def image_contrast(image):
     min_intensity=np.min(image)
     max_intensity=np.max(image)
     contrast=(max_intensity-min_intensity)/(max_intensity+min_intensity)
     return contrast

# 图像熵，图像熵是图像信息内容的度量，可以反应图像的复杂度和纹理信息

def image_entropy(image):
    # gitpilot的代码
    entropy=cv2.calcHist([image],[0],None,[256],[0,256])
    entropy=np.squeeze(entropy)
    entropy=entropy[entropy>0]
    ent=-np.sum(entropy*np.log2(entropy))

    # gpt的代码
    # histogram,_=np.histogram(image,bins=256,range=(0,1))
    # histo_norm=histogram/np.sum(histogram)
    # ent=entropy(histo_norm)
    return ent

# 颜色饱和度，色彩饱和度可以反应图像的生动度。某些情况下，饱和度低可能意味着图像模糊
def color_saturation(image_path):
     image=cv2.imread(image_path)
     hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
     saturation=hsv_image[:,:,1]
     mean_saturation=np.mean(saturation)
     return mean_saturation

# 图像亮度，过亮或过暗的图像可能会显得模糊
def image_brightness(image):
    image=cv2.imread(image)
    hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    brightness=hsv_image[:,:,2]
    mean_brightness=np.mean(brightness)
    return mean_brightness

# 下面的方法来自于知乎，上面的方法来自于gpt4
#SMD梯度函数计算
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

#SMD2梯度函数计算

def SMD2(img):
    """
    :param img: ndarray 二维灰度图像
    :return: float 图像越清晰越大
    """
    diff_x = np.abs(img[:-1, :-1] - img[1:, :-1])
    diff_y = np.abs(img[:-1, :-1] - img[:-1, 1:])

    out = np.sum(diff_x * diff_y)

    return float(out)


#brenner梯度函数计算
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

#方差函数计算
def variance(img):
    """
    :param img: ndarray 二维灰度图像
    :return: float 图像越清晰越大
    """
    u = np.mean(img)
    out = np.sum((img - u)**2)

    return float(out)


#energy函数计算
def energy(img):
    """
    :param img: ndarray 二维灰度图像
    :return: float 图像越清晰越大
    """
    diff_x = np.square(img[1:, :-1] - img[:-1, :-1])
    diff_y = np.square(img[:-1, 1:] - img[:-1, :-1])

    out = np.sum(diff_x * diff_y)

    return float(out)


#Vollath函数计算
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

#entropy函数计算
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out