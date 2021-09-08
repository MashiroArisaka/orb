# -*- codeing = utf-8 -*-

import cv2
import time
import os
from pathlib import Path
from sys import argv, exit
import numpy as np


img1_path = r'./IMGV/11.jpg' #需要搜图的图片
dir_name = './IMG/' #本地图片数据目录

result_file = 'result_file.md' #结果md文件名
result_list = {}
Filelist = []


#自定义计算两个图片相似度函数
def img_similarity(img1_path,img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    try:
        # 读取图片
        #img1 = cv2.imread(raw_data1, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imdecode((np.fromfile(img1_path, dtype=np.uint8)), cv2.IMREAD_GRAYSCALE)

        #img2 = cv2.imread(raw_data2, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode((np.fromfile(img2_path, dtype=np.uint8)), cv2.IMREAD_GRAYSCALE)

        # 初始化ORB检测器
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        print(f'{len(good)} / {len(matches)}')

        if len(matches) > 0 :
            similary = len(good) / len(matches)
        else:
            similary = 0

        #print("两张图片相似度为:%s" % similary)
        result_list[img1_path] = similary
        #result_list[img2_path] = similary
        print(f"{img1_path} | {img2_path} | %s" % similary)

        return similary

    except:
        result_list[img1_path] = 0
        print(f"{img1_path} | {img2_path} | error-0")
        return '0'


#遍历获取(子)文件夹所有图片文件
def get_filelist(dir):
    Filelist = []
    for home, dirs, files in os.walk(dir_name):
        for filename in files:
            if '.jpeg' in filename or '.jpg' in filename or '.png' in filename:
                Filelist.append(os.path.join(home, filename)) # 文件名列表，包含完整路径
                #Filelist.append(filename) # 文件名列表，只包含文件名
    return Filelist

if __name__ == '__main__':
    #'''
    Filelist = get_filelist(dir) #图片文件列表

    start = time.time()
    for file in Filelist: #开始检测
        similary = img_similarity(file, img1_path)
        #similary = img_similarity(img1_path, file)
    end = time.time()
    print (f'计算耗时{end - start}秒')

    #print(result_list)

    #先w清空md文件，并写入固定表格信息
    with open(result_file, 'w', encoding="utf-8") as f:
        f.write(f'|匹配图|相似度|<img src="{img1_path}"  width=150><br>搜图↑|\n| :-: | :-: | :-: |\n')
    
    #进行相似度排序
    for k in sorted(result_list, key=result_list.__getitem__, reverse=True):
        #print(f'{k} | {result_list[k]}')
        #print(f'| <img src="{k}"  width=150> | {result_list[k]} | {k} |')
        #写入结果到md文件
        with open(result_file, 'a', encoding="utf-8") as f:
            f.write(f'| <img src="{k}"  width=250> | {result_list[k]} | {k} |\n')
    #'''

    '''
    for filename in Path(dir_name).glob("**/*"):  # 遍历所有子目录子文件
        if os.path.isfile(filename):  # 判断是否为文件
            if '.jpeg' in str(filename) or '.jpg' in str(filename) or '.png' in str(filename):
                #print(filename)
                Filelist.append(filename)

    for file in Filelist: #开始检测
        similary = img_similarity(str(file), img1_path)
        #similary = img_similarity(img1_path, file)
    '''

    '''
    file_list = os.listdir(dir_name)
    #print(file_list)
    start = time.time()
    for i in file_list:
        similary = img_similarity(f'{dir_name}{i}', img1_path)
        #similary = img_similarity(img1_path, f'{dir_name}{i}')
    end = time.time()
    print (f'计算耗时{end - start}秒')
    '''
