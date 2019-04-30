# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:59:35 2018

@author: Unclered
"""
# assign子函数用于划分训练集和测试集。increment子函数用于序贯增加训练集的数量
# 修改主函数的dataBase和name，用以获取不同的数据集

import scipy.io as scio  #载入mat格式数据
import numpy as np
import random
'''
startClass：起始的类别数。也就是总类别数
trainNum：每一类准备分出多少张来作为序贯的最终数量
numGallery：gallery的类别数。如果不考虑代价敏感的话，直接与stratClass一样大并且numImpostor为0。
images：所有图片（包含训练集和测试集），之后本程序会划分训练集和测试集。labels也一样
'''
def assign(inputSize, startClass, trainNum, numGallery, numImpostor, images, labels):
    numClass = numGallery + numImpostor  # 剩余类别数
    allImages, allLabels = classify(inputSize, startClass, numGallery, numImpostor, images, labels)  # 获取数据和标记
    # 切割数据和标记
    trainImages, trainLabels, testImages, testLabels = cut(inputSize, trainNum, allImages, allLabels, numClass)
    return trainImages,testImages,trainLabels,testLabels
 
# 随机获得正类和负类，前numGallery为正类，后numImpostor为负类
def classify(inputSize,numClass,numGallery,numImpostor,images,labels):  # 将数据分为正类和负类
    classL = range(1,numClass+1)
    galleryL = random.sample(classL, numGallery)
    impostors = list(set(classL) - set(galleryL))
    impostorL = random.sample(impostors,numImpostor)
    #分配数据
    number = images.shape[0]  # 图片总数
    num_each = int(number/numClass)  # 每个人的训练张数
    # 初始化数据
    galleryImages = np.zeros((numGallery * num_each, inputSize))
    galleryLabels = np.zeros((numGallery * num_each, 1))
    impostorImages = np.zeros((numImpostor * num_each, inputSize))
    impostorLabels = np.zeros((numImpostor * num_each, 1))
    countG = 0  # 正类计数
    countI = 0  # 负类计数
    #获取数据和标记
    for i in range(numClass):
        start = i * num_each
        iterImages = images[start:start + num_each]  # 按顺序读取每一类数据
        np.random.shuffle(iterImages)  # 将数据顺序打乱。类标记由于是相同的，因此不用管
        #iterLabels = labels[start:start + num_each]
        if (i+1) in galleryL:
            galleryImages[countG * num_each:(countG+1) * num_each] = iterImages
            galleryLabels[countG * num_each:(countG+1) * num_each] = (countG+1) * np.ones((num_each, 1))  # 新标记
            #galleryLabels[countG * num_each:(countG+1) * num_each] = iterLabels  # 保留标记
            countG += 1
        elif (i+1) in impostorL:
            impostorImages[countI * num_each:(countI+1) * num_each] = iterImages
            impostorLabels[countI * num_each:(countI+1) * num_each] = (countI+numGallery+1) * np.ones((num_each, 1))#新标记
            #impostorLabels[countI * num_each:(countI+1) * num_each] = iterLabels  # 保留标记
            countI += 1
    # 返回图像数据和标记        
    allImages = np.vstack((galleryImages,impostorImages))  # 将数据合并
    allLabels = np.vstack((galleryLabels,impostorLabels))  # 将标记合并
    return allImages, allLabels

def cut(inputSize,trainNum, images, labels, numClass):
    # 将分好的总数据，分割成训练集和测试集
    number = labels.shape[0]
    num_each = int(number/numClass)
    testNum = num_each - trainNum
    # 将类别变为所需类型
    initLabels = np.tile(labels,(1,numClass))#复制
    classL = range(1,numClass+1)
    tempLabels = np.tile(np.array(classL),(number,1))
    Labels = initLabels==tempLabels
    Labels = Labels.astype(np.float)  # 将矩阵的布尔型改为float型
    #初始化数据
    trainImages = np.zeros((numClass * trainNum, inputSize))
    trainLabels = np.zeros((numClass * trainNum, numClass))
    testImages = np.zeros((numClass * testNum, inputSize))
    testLabels = np.zeros((numClass * testNum, numClass))
    for i in range(numClass):
        start = i * num_each
        startTrain = i * trainNum
        startTest = i * testNum
        trainImages[startTrain : startTrain+trainNum] = images[start : start+trainNum]#读取每一类数据
        trainLabels[startTrain : startTrain+trainNum] = Labels[start : start+trainNum]
        testImages[startTest : startTest+testNum] = images[start+trainNum : start+num_each]
        testLabels[startTest  :startTest+testNum] = Labels[start+trainNum : start+num_each]
    return trainImages, trainLabels, testImages, testLabels

def acceptRest(allTrainImages,allTrainLabels,numClass,numTrain=1,inputSize=2500):
    '''设定初始训练集和剩余集，初始训练集张数暂定为1'''
    numAll = allTrainLabels.shape[0]
    numEach = int(numAll/numClass)
    numRest= numEach - numTrain
    allTrain = numClass * numTrain
    allRest = numClass * numRest
    trainImages = np.zeros([allTrain, inputSize])
    trainLabels = np.zeros([allTrain, numClass])
    restImages = np.zeros([allRest, inputSize])
    restLabels = np.zeros([allRest, numClass])
    for i in range(numClass):
        startTrain = i * numTrain
        startRest = i * numRest 
        startAll = i * numEach 
        trainImages[startTrain:startTrain+numTrain,:] = allTrainImages[startAll:startAll+numTrain,:]
        trainLabels[startTrain:startTrain+numTrain,:] = allTrainLabels[startAll:startAll+numTrain,:]
        restImages[startRest:startRest+numRest,:] = allTrainImages[startAll+numTrain:startAll+numTrain+numRest,:]
        restLabels[startRest:startRest+numRest,:] = allTrainLabels[startAll+numTrain:startAll+numTrain+numRest,:]
    return trainImages, trainLabels, restImages, restLabels
    
def acceptVerify(allTestImages,allTestLabels,numClass,numVerify,inputSize=2500):
    # 从测试集中将每类最后几张作为验证集
    if numVerify != 0:
        numAll = allTestLabels.shape[0]
        numEach = int(numAll/numClass)
        numTest = numEach - numVerify#此处测试集数量可手动定义，不影响后面的代码
        allTest = numClass * numTest
        allVerify = numClass * numVerify
        testImages = np.zeros([allTest, inputSize])
        testLabels = np.zeros([allTest, numClass])
        verifyImages = np.zeros([allVerify, inputSize])
        verifyLabels = np.zeros([allVerify, numClass])
        for i in range(numClass):
            startTest = i * numTest
            startVerify = i * numVerify 
            startAll = i * numEach 
            testImages[startTest:startTest+numTest,:] = allTestImages[startAll:startAll+numTest,:]
            testLabels[startTest:startTest+numTest,:] = allTestLabels[startAll:startAll+numTest,:]
            verifyImages[startVerify:startVerify+numVerify,:] = allTestImages[startAll+numTest:startAll+numTest+numVerify,:]
            verifyLabels[startVerify:startVerify+numVerify,:] = allTestLabels[startAll+numTest:startAll+numTest+numVerify,:]
    else:
        testImages = allTestImages
        testLabels = allTestLabels
        verifyImages = np.zeros([1,inputSize])
        verifyLabels = np.zeros([1,numClass])
            
    return testImages, testLabels, verifyImages, verifyLabels



# 如果没有主函数调用该函数则此部分会运行，用于制作需要的数据集
if __name__=='__main__':
    inputSize = 2500
    dataBase = 0  # 选择数据集
    if dataBase == 0:
        numClass = 38
        trainNum = 24  # 训练集张数
        data_path = 'Face/allEYaleB_50'
        name = 'Data/EYaleB10.mat'
    elif dataBase == 1:
        numClass = 100 
        trainNum = 7
        data_path = 'Face/allAR_50'
        name = 'Data/AR1.mat'
    elif dataBase == 2:
        numClass = 68 
        trainNum = 8
        data_path = 'Face/allPIE68_50'
        name = 'Data/PIE1.mat'
    increNum = 25
    numGallery = 20
    numImpostor = 15
    data = scio.loadmat(data_path)
    images = data.get('images')  # images是文件中数据的名称
    labels = data.get('labels')
    trainImages, testImages, trainLabels, testLabels = assign(inputSize, numClass, trainNum,
                                                           numGallery, numImpostor, images, labels)
    scio.savemat(name, {'trainImages': trainImages, 'trainLabels': trainLabels,
                                  'testImages': testImages, 'testLabels': testLabels})


