# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:36:10 2019

@author: Unclered
"""
# 先运行assignData获取数据集
# 两层隐层，之后微调，调用的是stackedSAE子函数
# 选择增量集的方法和人脸数据集由methodSelect和dataSetSelect两个参数决定
# dataSetSelect是选择人脸数据集的参数。0代表YaleB，1代表PIE
# methodSelect是选择增量集的参数。0是随机, 1是最小边际, 2是装袋熵

import numpy as np
from openpyxl import Workbook
import scipy.io as scio    # 载入mat格式数据

import assignData
import activeLearning 
import stackedSAE
import sequential
import computeCost

dataSetList = ['EYaleB', 'PIE', 'AR']
# dataSetSelect = 1
numGallery = 20            # gallery类个数
numImpostor = 15           # impostor类个数 
# 计算最终类别数
numClass = numGallery + numImpostor
# 几乎不变的超参数
inputSize = 2500           # 输入维度
hiddenSizeOne = 1000       # 第一隐层维度，
hiddenSizeTwo = 400        # 第二隐层维度，也即softmax输入维度
saeLearningRate = 5e-3     # SAE学习速率
saeIterNum = 2000          # SAE迭代次数
saeLambda = 1e-5           # 权值惩罚参数，好像1e-5更好，之后可以再改
beta = 0.1                 # 稀疏性权重
rho = 0.1                  # 稀疏性参数
softLearningRate = 1e-2    # softmax学习速率
softIterNum = 1500         # softmax迭代次数
softLambda = 3e-5          # softmax权重衰减系数
fineTuneIterNum = 2000     # 微调次数
fineTuneLearningRate = 5e-3  # 微调学习率
numStep = 24               #决策步骤数
sizePara = [inputSize, hiddenSizeOne, hiddenSizeTwo]
saePara = [saeLearningRate, saeIterNum, saeLambda, beta, rho]
softPara = [softLearningRate, softIterNum, softLambda, fineTuneIterNum, fineTuneLearningRate]
# 10次取平均
for cycle in range(1, 2):
    for dataSetSelect in range(2):
        dataSet = dataSetList[dataSetSelect]
        if dataSetSelect == 0:
            numTrain = 24              # 训练集数量  
            numVerify = 10             # 验证集数量
            numIncre = 25              # 每次增量数
            numInit = 5                # 初始每类数
        else:
            numTrain = 8               # 训练集数量  
            numVerify = 0              # 验证集数量
            numIncre = 8               # 每次增量数
            numInit = 2                # 初始每类数
            '''
            AR的参数
            numTrain = 7               # 训练集数量  
            numVerify = 0              # 验证集数量
            numIncre = 7               # 每次增量数
            numInit = 2                # 初始每类数
            '''
        for methodSelect in range(2):
            # 读取数据
            useNumber = numInit * numClass
            data_path = 'Data/' + dataSet + str(cycle) + '.mat'
            data = scio.loadmat(data_path)
            allTrainImages = data.get('trainImages') 
            allTrainLabels = data.get('trainLabels')
            allTestImages = data.get('testImages')
            allTestLabels = data.get('testLabels')
            # 切割出初始训练集和剩余集
            trainImages, trainLabels, restImages, restLabels = \
            assignData.acceptRest(allTrainImages, allTrainLabels, numClass, numTrain=numInit)
            # 切割出测试集和验证集
            testImages, testLabels, verifyImages, verifyLabels = \
            assignData.acceptVerify(allTestImages, allTestLabels, numClass, numVerify)
            # 构建输出表格
            workbook = Workbook()
            booksheet = workbook.active  # 获取当前活跃的sheet,默认是第一个sheet
            booksheet.append(['twoCost', 'twoAcc', 'twoAccIm', 'N_GI2', 'N_IG2', 
                              'threeCost', 'threeAcc', 'threeAccIm',
                              'N_GI3','N_IG3','N_GB','N_IB','runTime', 'useData'])
            for step in range(numStep):
                # 第一步， 获取数据和标记
                # 用assign函数切割数据
                increStep = step + 1  # 每步决策用的每类图片数（三支决策时循环得到）
                print()  # 打印一个空行，方便查看结果
                print('Number of experiment is %d. The dataSet is %s, The method is %d. Number of train is %d'
                      %(cycle, dataSet, methodSelect, increStep))
                
                if methodSelect == 0:
                    if increStep != 1:
                        useNumber = numIncre
                        increImages, increLabels, restImages, restLabels = \
                        activeLearning.randomSelect(restImages, restLabels, numIncre)
                        trainImages = np.vstack([trainImages, increImages])
                        trainLabels = np.vstack([trainLabels, increLabels])
                    # 第二步， 用SAE提取特征, 将训练集的隐层输出即可
                    trainProb, twoTestPred, testProb, restPred, restProb, runTime = \
                    stackedSAE.SAE(numClass, sizePara, saePara, softPara,
                                   trainImages, trainLabels, testImages, testLabels, restImages, show=1000)
                if methodSelect == 1:
                    if increStep != 1:
                        useNumber = numIncre
                        increImages, increLabels, restImages, restLabels, diffValueList = \
                        activeLearning.CSmarginSample(restImages, restLabels, restProb, numIncre)
                        trainImages = np.vstack([trainImages, increImages])
                        trainLabels = np.vstack([trainLabels, increLabels])
                    # 第二步， 用SAE提取特征, 将训练集的隐层输出即可
                    trainProb, twoTestPred, testProb, restPred, restProb, runTime = \
                    stackedSAE.SAE(numClass, sizePara, saePara, softPara,
                                   trainImages, trainLabels, testImages, testLabels, restImages, show=1000)
                # 该方法效果不佳
                elif methodSelect == 2:
                    #装袋熵方法
                    restProb2, restProb3 = 0, 0
                    state = np.random.get_state()
                    np.random.shuffle(restImages)
                    np.random.set_state(state)
                    np.random.shuffle(restLabels)  
                    if increStep != 1:
                        increImages, increLabels, restImages, restLabels = \
                        activeLearning.CSEQB(restImages, restLabels, restProb, restProb2, restProb3, numIncre)
                        trainImages = np.vstack([trainImages, increImages])
                        trainLabels = np.vstack([trainLabels, increLabels])   
                    # 第二步， 用SAE提取特征, 将训练集的隐层输出即可
                    trainProb, twoTestPred, testProb, restPred, restProb, restProb2, restProb3, runTime = \
                    stackedSAE.threeSAE(numClass, sizePara, saePara, softPara,
                                        trainImages, trainLabels, testImages, testLabels, restImages, show=1000)
            
                restLabelsList = np.argmax(restLabels, axis=1)  # 最大值索引，也就是类别
                labelList = list(restLabelsList)
                restNumberList = [labelList.count(i) for i in range(numClass)]
                print(restNumberList)
                # 第四步，计算代价
                C_GB = 1
                C_IB = 2
                C_GI = 3
                C_IG = 12
                costList = [C_GB, C_IB, C_GI, C_IG]
                # twoPred为二支决策的结果，probability为概率值。由概率计算出三支决策的预测值threePred
                threePredTest = sequential.threeDecision(numGallery, numImpostor, testProb, costList)
                twomisCost, twoAcc, twoAccIm, twoNgi, twoNig = \
                computeCost.twoMisCost(numGallery, numImpostor, twoTestPred, costList)
                threemisCost, threeAcc, threeAccIm, threeNgi, threeNig, threeNgb, threeNib = \
                computeCost.threeMisCost(numGallery, numImpostor, threePredTest, costList)
                outList = [twomisCost, twoAcc, twoAccIm, twoNgi, twoNig, threemisCost, threeAcc, threeAccIm,
                           threeNgi, threeNig, threeNgb, threeNib, runTime, useNumber]
                booksheet.append(outList)
                print('twoCost is %d, N_GI is %d, N_IG is %d, Accuracy of 2WD is %.6f.'
                      %(twomisCost, twoNgi, twoNig, twoAcc))
                print('threeCost is %d, N_GB is %d, N_IB is %d, Accuracy of 3WD is %.6f.'
                      %(threemisCost, threeNgb, threeNib, threeAcc))
                        
            #将结果储存为Excel
            methodList = ['randomSelect', 'CSmarginSample', 'CSEQB']
            saveName = 'Results/' + methodList[methodSelect] + '/' + dataSet + '_0130_' + str(cycle) + '.xlsx'
            workbook.save(saveName)
    