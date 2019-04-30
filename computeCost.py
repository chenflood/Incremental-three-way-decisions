# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:44:32 2018

@author: Unclered
"""

def twoMisCost(numGallery,numImpostor,prediction,costList):
    number = prediction.shape[0]
    num_each = int(number/(numGallery+numImpostor))
    # labels = [i//num_each +1 for i in range(number)]
    pred = list(prediction)
    C_GI = costList[2]
    C_IG = costList[3]
    # 计算代价和计数
    cost, N_GI, N_IG, T_G, T_I = 0, 0, 0, 0, 0
    for i in range(numGallery * num_each):
        if pred[i] >= numGallery:
            cost += C_GI
            N_GI += 1
        elif pred[i] == int(i/num_each):
            T_G += 1
    for i in range(numGallery * num_each, number):
        if pred[i] < numGallery:
            cost += C_IG
            N_IG += 1
        elif pred[i] == int(i/num_each):
            T_I += 1
    twoAcc = (T_G + T_I)/number
    twoAccIm = T_I/(numImpostor * num_each)
    return cost, twoAcc, twoAccIm, N_GI, N_IG
    
def threeMisCost(numGallery,numImpostor,prediction,costList,costWeight=0):
    #prediction是一个列表
    number = len(prediction)
    num_each = int(number/(numGallery+numImpostor))
    C_GB, C_IB, C_GI, C_IG = costList
    # 计算代价和计数
    cost, N_GI, N_IG, N_GB, N_IB, T_G, T_I = 0, 0, 0, 0, 0, 0, 0
    for i in range(numGallery * num_each):
        if prediction[i] >= numGallery:
            cost += C_GI
            N_GI += 1
        elif prediction[i] == -1:
            cost += C_GB
            N_GB += 1
        elif prediction[i] == int(i/num_each):
            T_G += 1
    for i in range(numGallery * num_each, number):
        if prediction[i] == -1:
            cost += C_IB
            N_IB += 1
        elif prediction[i] < numGallery:
            cost += C_IG
            N_IG += 1
        elif prediction[i] == int(i/num_each):
            T_I += 1
    threeAcc = (T_G + T_I)/number
    threeAccIm = T_I/(numImpostor * num_each)
    cost += costWeight * (N_GB + N_IB)
    return cost ,threeAcc, threeAccIm, N_GI, N_IG, N_GB, N_IB
    