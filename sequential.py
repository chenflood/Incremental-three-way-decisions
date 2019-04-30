import numpy as np

def threeDecision(numGallery, numImpostor, probability, costList):
    #前numGallery类为gallery，后numImpostor类为impostor
    number = probability.shape[0]#样本数量
    '''
    numClass = numGallery + numImpostor
    eachNum = int(number/numClass)
    num_gallery = eachNum * numGallery
    num_impostor = eachNum * numImpostor
    '''
    labels = np.argmax(probability,axis=1)
    # 先根据最大索引判断其属于哪一类，再去判断是否在边界域。而不是直接将概率加和
    C_GB, C_IB, C_GI, C_IG = costList
    alpha = (C_IG-C_IB)/(C_IG-C_IB+C_GB)  # 上边界
    beta = C_IB/(C_IB+C_GI-C_GB)  # 下边界
    # 获得的预测值最终不加一。也就是说，类别数从0开始
    prediction = []
    for i in range(number):
        if labels[i] < numGallery:
            if probability[i][labels[i]] >= alpha:
                prediction.append(labels[i])
            else:
                prediction.append(-1)
        else:
            if probability[i][labels[i]] > 1-beta:
                prediction.append(labels[i])
            else:
                prediction.append(-1)  # 与MATLAB不同，因为python索引从0开始，所以边界域类别为-1
    return prediction

def threeDecisionAdd(numGallery, numImpostor, probability, costList):
    # 前numGallery类为gallery，后numImpostor类为impostor
    number = probability.shape[0]#样本数量
    # 直接将类别概率加和
    gallery = probability[:, 0:numGallery]  # gallery类的概率值。之后计算预测类别会用到
    impostor = probability[:, -numImpostor:]  # impostor类的概率值。
    pg = gallery.sum(axis=1)  # gallery类的概率之和
    pi = impostor.sum(axis=1)  # impostor类的概率之和
    # 验证是否正确，只需要用pg+pi看看是否等于1。用np.add(a,b)和a+b都可以。经过验证没问题
    C_GB, C_IB, C_GI, C_IG = costList
    alpha = (C_IG-C_IB)/(C_IG-C_IB+C_GB)  # 上边界
    beta = C_IB/(C_IB+C_GI-C_GB)  # 下边界
    maxG = np.argmax(gallery,1)  # 判断为gallery类时的具体预测值
    maxI = np.argmax(impostor,1) + numGallery  # 判断为impostor类时的具体预测值
    # 预测
    prediction = []
    for i in range(number):
        if pg[i] >= alpha:
            prediction.append(maxG[i])
        elif pg[i] <= beta:
            prediction.append(maxI[i])
        else:
            prediction.append(-1)  # 与MATLAB不同，因为python索引从0开始，所以边界域类别为-1
    return prediction
			
			
