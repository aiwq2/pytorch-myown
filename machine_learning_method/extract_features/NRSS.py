# 利用Reblur 二次模糊图像与原图像的差异来评估图像的清晰度
# 详情可见知乎https://zhuanlan.zhihu.com/p/97024018


#encoding=utf-8
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def gauseBlur(img):
    img_Guassian = cv2.GaussianBlur(img,(7,7),0)
    return img_Guassian

def loadImage(filepath):
    img = cv2.imread(filepath, 0)  ##   读入灰度图
    return img
def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
def saveImage(path, img):
    cv2.imwrite(path, img)
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst
def getBlock(G,Gr):
    (h, w) = G.shape
    G_blk_list = []
    Gr_blk_list = []
    sp = 6
    for i in range(sp):
        for j in range(sp):
            G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            G_blk_list.append(G_blk)
            Gr_blk_list.append(Gr_blk)
    sum = 0
    for i in range(sp*sp):
        mssim = structural_similarity(G_blk_list[i], Gr_blk_list[i])
        sum = mssim + sum
    nrss = 1-sum/(sp*sp*1.0)
    # nrss值越大说明越模糊
    return nrss
def NRSS(path):
    image = loadImage(path)
    #高斯滤波
    Ir = gauseBlur(image)
    G = sobel(image)
    Gr = sobel(Ir)
    blocksize = 8
    ## 获取块信息
    nrss=getBlock(G, Gr)
    return nrss