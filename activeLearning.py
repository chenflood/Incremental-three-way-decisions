# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:04:40 2019

@author: Unclered
"""
# 用主动学习的方式来增加数据打上标记

import numpy as np
import random
from math import log

def randomSelect(restImages,restLabels,numIncre,inputSize=2500,numClass=35):
    # 随机选取一定数量的样本加上标记
    number = restImages.shape[0]
    indexList = range(number)
    selectIndex = random.sample(indexList, numIncre)  # 从indexList中获得索引不相同的numIncre个元素
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels

def leastProb(restImages,restLabels,restProb,numIncre,inputSize=2500,numClass=35):
    # 获得最大概率与次大概率的差值，差值最小的就认为是最没有信心的。就加入进来
    number = restImages.shape[0]
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    maxValueList = []
    maxIndex = np.argmax(restProb, axis=1)#最大值索引
    for i in range(number):
        maxValueList.append(restProb[i,maxIndex[i]])
    selectList = np.argsort(maxValueList)#从小到大排序
    selectIndex = selectList[:numIncre]#找寻最小的几个数                                               
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels, maxValueList

def entropySelect(restImages,restLabels,restProb,numIncre,inputSize=2500,numClass=35):
    # 用熵值来选取需要打标记的数据
    number = restImages.shape[0]
    entropyList = []
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    for i in range(number):
        # 结束后获得一个信息熵列表
        entropy = 0
        for j in range(numClass):
            # 获得每一个数据的信息熵
            entropy += restProb[i,j] * log(restProb[i,j], 2)
        entropyList.append(entropy)                                                 
    selectList = np.argsort(entropyList)  # 从小到大排序
    selectIndex = selectList[:numIncre]  # 找寻最小的几个数                                               
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels

    
def marginSample(restImages,restLabels,restProb,numIncre,inputSize=2500,numClass=35):
    # 获得最大概率与次大概率的差值，差值最小的就认为是最没有信心的。就加入进来
    number = restImages.shape[0]
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    maxValueList = []
    maxIndex = np.argmax(restProb, axis=1)#最大值索引
    for i in range(number):
        maxValueList.append(restProb[i,maxIndex[i]])
        restProb[i,maxIndex[i]] = 0
    nextIndex = np.argmax(restProb, axis=1)  # 次大值索引
    diffValueList = [maxValueList[i] - restProb[i,nextIndex[i]] for i in range(number)]
    selectList = np.argsort(diffValueList)  # 从小到大排序
    selectIndex = selectList[:numIncre]  # 找寻最小的几个数                                               
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels, diffValueList

def CSmarginSample(restImages,restLabels,restProb,numIncre,numGallery=25,inputSize=2500,numClass=35):
    # 获得最大概率与次大概率的差值，差值最小的就认为是最没有信心的。就加入进来
    # 对差值乘权重。同类权重为1，gi权重为2，ig权重为5？或者gi与ig权重相同会更好？
    # 毕竟也不一定是分类正确，只要可能分类错误就加倍更好
    # 信息熵的话，目测也有代价敏感信息熵？
    number = restImages.shape[0]
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    maxValueList = []
    maxIndex = np.argmax(restProb, axis=1)  # 最大值索引，也就是类别
    for i in range(number):
        maxValueList.append(restProb[i,maxIndex[i]])
        restProb[i,maxIndex[i]] = 0
    nextIndex = np.argmax(restProb, axis=1)  # 次大值索引，也就是次大值类别
    diffValueList = [maxValueList[i] - restProb[i,nextIndex[i]] for i in range(number)]
    for i in range(number):
        if maxIndex[i] < numGallery and nextIndex[i] >= numGallery:
            diffValueList[i] = diffValueList[i] * 0.5  # gi的代价
        elif maxIndex[i] >= numGallery and nextIndex[i] < numGallery:
            diffValueList[i] = diffValueList[i] * 0.25  # ig的代价
    selectList = np.argsort(diffValueList)  # 从小到大排序
    selectIndex = selectList[:numIncre]  # 找寻最小的几个数                                               
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels, diffValueList
    
def EQB(restImages,restLabels,restProb1,restProb2,restProb3,numIncre,numGallery=25,inputSize=2500,numClass=35): 
    number = restImages.shape[0]
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    label1 = np.argmax(restProb1, axis=1)  # 最大值索引，也就是类别
    label2 = np.argmax(restProb2, axis=1)  # 最大值索引，也就是类别
    label3 = np.argmax(restProb3, axis=1)  # 最大值索引，也就是类别
    entropy = []  # 依次计算每个样本的交叉熵，加负号这样就可以取最小
    for i in range(number):
        N={}
        Hbag = 0
        label = [label1[i],label2[i],label3[i]]
        # 制作类别列表
        for j in label:
            if j not in N:
                N[j] = 1
            else:
                N[j] += 1
        # 计算熵
        for key,value in N.items():
            P = value/3
            Hbag += P * log(P,10)
        entropy.append(Hbag)
    selectList = np.argsort(entropy)  # 从小到大排序
    selectIndex = selectList[:numIncre]  # 找寻最小的几个数 
    # print(selectIndex)                                          
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels

def CSEQB(restImages,restLabels,restProb1,restProb2,restProb3,numIncre,numGallery=25,inputSize=2500,numClass=35): 
    number = restImages.shape[0]
    increImages = np.zeros([numIncre,inputSize])
    increLabels = np.zeros([numIncre,numClass])
    label1 = np.argmax(restProb1, axis=1)  # 最大值索引，也就是类别
    label2 = np.argmax(restProb2, axis=1)  # 最大值索引，也就是类别
    label3 = np.argmax(restProb3, axis=1)  # 最大值索引，也就是类别
    entropy = []  # 依次计算每个样本的交叉熵，加负号这样就可以取最小
    for i in range(number):
        key=[]
        value=[]
        Hbag = 0
        label = [label1[i],label2[i],label3[i]]
        # 制作类别列表
        for j in label:
            if j not in key:
                key.append(j)
                value.append(1)
            else:
                keyIndex=key.index(j)
                value[keyIndex] = value[keyIndex] + 1
        keyLabel = [m-numGallery+0.5 for m in key]
        # 计算熵
        if len(key) == 3:
            if keyLabel[0] * keyLabel[1] * keyLabel[2] < 0:
                if keyLabel[0]<0 and keyLabel[1]<0 and keyLabel[2]<0:
                    Hbag=4
                else:
                    Hbag=1
            else:
                if keyLabel[0]>0 and keyLabel[1]>0 and keyLabel[2]>0:
                    Hbag=3
                else:
                    Hbag=2
        elif len(key) == 2:
            if keyLabel[0] * keyLabel[1] <  0:
                Hbag=5
            else:
                Hbag=6
        elif len(key) == 1:
            Hbag=7
        entropy.append(Hbag)
    selectList = np.argsort(entropy)  # 从小到大排序
    print(entropy[:25])
    selectIndex = selectList[:numIncre]  # 找寻最小的几个数 
    # print(selectIndex)                                          
    for i in range(numIncre):
        increImages[i,:] = restImages[selectIndex[i],:]
        increLabels[i,:] = restLabels[selectIndex[i],:]
    minIndex = np.argsort(selectIndex)
    maxIndex = minIndex[::-1]
    # 根据索引删除数据一定要倒序删除。否则数据量减小，索引不再是原来的索引了
    for i in range(numIncre):
        select = maxIndex[i]
        restImages = np.delete(restImages,selectIndex[select],axis=0)
        restLabels = np.delete(restLabels,selectIndex[select],axis=0)
    return increImages, increLabels, restImages, restLabels
   
    
    
    
    
    
    
    
    
    
    
    