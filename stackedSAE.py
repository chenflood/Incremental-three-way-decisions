# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:58:22 2019

@author: Unclered
"""

import numpy as np
import tensorflow as tf
import time

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),minval = low, maxval = high,dtype = tf.float32)  # 此处高版本可能报错

class SparseAutoencoder(object):
    def __init__(self, inputSize, hiddenSize, saeLambda, beta, rho, transfer_function, optimizer):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.transfer = transfer_function
        self.saeLambda = saeLambda
        self.beta = beta
        self.rho = rho
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # 定义模型，也就是输入层，隐含层，输出层以及之间的映射矩阵
        self.x = tf.placeholder(tf.float32, [None, self.inputSize])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']),self.weights['b1']))
        self.reconstruction = self.transfer(tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']))
        # 定义损失函数，这里我们使用均方差，因为下面的激活函数选择的是恒等
        error_cost = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0),axis=0))
        weight_decay = tf.nn.l2_loss(self.weights['w1']) + tf.nn.l2_loss(self.weights['w2'])
        # 计算KL散度
        rho_hat = tf.reduce_mean(self.hidden, axis=0)
        KL = self.rho * tf.log(self.rho) - self.rho * tf.log(rho_hat) + \
               (1 - self.rho) * tf.log(1 - self.rho) - (1 - self.rho) * tf.log(1 - rho_hat)
        self.cost = 0.5 * error_cost + 0.5 * self.saeLambda * weight_decay + self.beta * tf.reduce_sum(KL)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)   
        
    def _initialize_weights(self):
        # 字典类型
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.inputSize, self.hiddenSize))
        all_weights['b1'] = tf.Variable(tf.zeros([self.hiddenSize], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(xavier_init(self.hiddenSize, self.inputSize))
        all_weights['b2'] = tf.Variable(tf.zeros([self.inputSize], dtype=tf.float32))
        return all_weights
    
    def partial_fit(self, X):  # 执行损失＋执行一步训练
        cost, opt = self.sess.run((self.cost, self.optimizer),feed_dict = {self.x: X})
        return cost
        
    def calc_total_cost(self, X):  # 只执行损失
        return self.sess.run(self.cost, feed_dict = {self.x: X})
    
    def transform(self,X):  # 为了看隐含层输出结果
        return self.sess.run(self.hidden, feed_dict = {self.x: X})

    def generate(self, hidden = None):  # 将隐含层输出的高阶特征还原为原始数据
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

    def reconstruct(self, X):  # transform + generator
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        W1 = self.sess.run(self.weights['w1'])
        W2 = self.sess.run(self.weights['w2'])
        return W1, W2

    def getBiases(self):
        b1 = self.sess.run(self.weights['b1'])
        b2 = self.sess.run(self.weights['b2'])
        return b1, b2
    
    def sessClose(self):
        self.sess.close()


def trainSoftmax(softSize,numClass,learningRate,iterNum,trainImages,trainLabels,weight=0,softStop=0,show=100):
    # 第一步，定义算法公式
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, softSize])  # 构建占位符，None表示样本的数量可以是任意的 
    W = tf.Variable(tf.random_normal([softSize, numClass]))  # 构建一个变量，代表权重矩阵，初始化为0
    b = tf.Variable(tf.zeros([numClass]))  # 构建一个变量，代表偏置，初始化为0
    y = tf.nn.softmax(tf.matmul(x, W) + b)  # 构建了一个softmax的模型
    
    # 第二步，定义损失函数，选定优化器，并指定优化器优化损失函数
    y_ = tf.placeholder(tf.float32, [None, numClass])
    # 交叉熵损失函数，注意因为有sum，所以只有一个部分。
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
    # 使用梯度下降法最小化cross_entropy损失函数
    weight_decay = 0.5 * tf.reduce_sum(tf.pow(W,2))
    # weight_decay = tf.nn.l2_loss(W)
    error_loss = cross_entropy + weight * weight_decay
    train_step = tf.train.AdamOptimizer(learningRate).minimize(error_loss)

    # 第三步，用不同方法对数据进行迭代训练
    tf.global_variables_initializer().run()
    start = time.time()
    endEpoch = iterNum
    for i in range(iterNum):  # 迭代次数
        _, cost = sess.run([train_step, error_loss], {x: trainImages, y_: trainLabels})
        if cost < softStop:
            endEpoch = i
            break
    print("Softmax Epoch:", '%04d' % (endEpoch), "cost=", "{:.9f}".format(cost))
    end = time.time()
    # 第四步，保存参数
    W_soft = sess.run(W, {x: trainImages, y_: trainLabels})
    b_soft = sess.run(b, {x: trainImages, y_: trainLabels})    
    runTime = end-start
    # print('Running time: %s Seconds'%(runTime))
    sess.close()
    
    return runTime, W_soft, b_soft


def fineTune(trainImages, trainLabels, testImages, testLabels, restImages, \
             weightPara, fineTuneLearningRate, fineTuneIterNum, show=500):
    #进行微调
    W1_encode, b1_encode, W2_encode, b2_encode, W_soft, b_soft = weightPara
    sess = tf.InteractiveSession()
    inputSize = W1_encode.shape[0]
    numClass = W_soft.shape[1]
    x = tf.placeholder(tf.float32, [None, inputSize])
    y_ = tf.placeholder(tf.float32, [None, numClass])
    W1 = tf.Variable(W1_encode)
    b1 = tf.Variable(b1_encode)
    W2 = tf.Variable(W2_encode)
    b2 = tf.Variable(b2_encode)
    W3 = tf.Variable(W_soft)
    b3 = tf.Variable(b_soft)
    # 构建网络
    hiddenOne = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
    hiddenTwo = tf.nn.sigmoid(tf.matmul(hiddenOne,W2) + b2)
    y_out = tf.nn.softmax(tf.matmul(hiddenTwo, W3) + b3)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_out),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(fineTuneLearningRate).minimize(cross_entropy)
    prediction = tf.argmax(y_out, 1)
    sess.run(tf.global_variables_initializer())
    # 训练
    start = time.time()
    for i in range(fineTuneIterNum):	
        _, cost = sess.run([train_step, cross_entropy], feed_dict={x:trainImages, y_: trainLabels})	
    print("FineTune Epoch:", '%04d' % (fineTuneIterNum), "cost=", "{:.9f}".format(cost))
    end = time.time()
    fineTuneTime = end - start
    trainProb = sess.run(y_out, feed_dict={x:trainImages})
    testPred = sess.run(prediction, feed_dict={x:testImages})
    testProb = sess.run(y_out, feed_dict={x:testImages})
    restPred = sess.run(prediction, feed_dict={x:restImages})
    restProb = sess.run(y_out, feed_dict={x:restImages})
    sess.close()
    
    return fineTuneTime, trainProb, testPred, testProb, restPred, restProb

    
def SAE(numClass, sizePara, saePara, softPara, \
        trainImages, trainLabels, testImages, testLabels, restImages, saeStop=0, show=500):
    # 还原出参数
    inputSize, hiddenSizeOne, hiddenSizeTwo = sizePara
    saeLearningRate, saeIterNum, saeLambda, beta, rho = saePara
    softLearningRate, softIterNum, softLambda, fineTuneIterNum, fineTuneLearningRate = softPara
    # 第一层隐层
    encoderOne = SparseAutoencoder(inputSize,hiddenSizeOne,saeLambda,beta,rho,transfer_function = tf.nn.sigmoid,
                                   optimizer = tf.train.AdamOptimizer(saeLearningRate))
    start1 = time.time()
    endEpoch = saeIterNum
    for i in range(saeIterNum):
        cost = encoderOne.partial_fit(trainImages)
        if cost < saeStop:
            endEpoch = i
            break
    print("One SAE Epoch:", '%04d' % (endEpoch), "cost=", "{:.9f}".format(cost))
    W1, W1_decode = encoderOne.getWeights()
    b1, b1_decode = encoderOne.getBiases()
    end1 = time.time()
    trainFeaturesOne = encoderOne.transform(trainImages)
    encoderOne.sessClose()
    # 第二层隐层
    encoderTwo = SparseAutoencoder(hiddenSizeOne,hiddenSizeTwo,saeLambda,beta,rho,transfer_function = tf.nn.sigmoid,
                                   optimizer = tf.train.AdamOptimizer(saeLearningRate))
    start2 = time.time()
    endEpoch = saeIterNum
    for i in range(saeIterNum):
        cost = encoderTwo.partial_fit(trainFeaturesOne)
        if cost < saeStop:
            endEpoch = i
            break
    print("Two SAE Epoch:", '%04d' % (endEpoch), "cost=", "{:.9f}".format(cost))
    W2, W2_decode = encoderTwo.getWeights()
    b2, b2_decode = encoderTwo.getBiases()
    end2 = time.time()
    trainFeaturesTwo = encoderTwo.transform(trainFeaturesOne)
    encoderTwo.sessClose()
    # softmax层
    softTime, W_soft, b_soft = trainSoftmax(hiddenSizeTwo,numClass,softLearningRate,softIterNum,\
                                            trainFeaturesTwo,trainLabels,weight=softLambda,show=show)
    # 进行微调并且输出预测值和概率值
    weightPara = [W1, b1, W2, b2, W_soft, b_soft]
    fineTuneTime,trainProb,testPred,testProb,restPred,restProb = \
    fineTune(trainImages,trainLabels,testImages,testLabels,restImages,\
             weightPara, fineTuneLearningRate,fineTuneIterNum)	
    tf.reset_default_graph()
    encodeOneTime = end1 - start1
    encodeTwoTime = end2 - start2
    runTime = encodeOneTime + encodeTwoTime + softTime + fineTuneTime
    print('Running time: %s Seconds'%runTime)
    return trainProb, testPred, testProb, restPred, restProb,runTime


def threeSAE(numClass, sizePara, saePara, softPara, \
             trainImages,trainLabels,testImages,testLabels,restImages,saeStop=0,show=500):
    # 还原出参数
    inputSize, hiddenSizeOne, hiddenSizeTwo = sizePara
    saeLearningRate, saeIterNum, saeLambda, beta, rho = saePara
    softLearningRate, softIterNum, softLambda, fineTuneIterNum, fineTuneLearningRate = softPara
    # 第一层隐层
    encoderOne = SparseAutoencoder(inputSize,hiddenSizeOne,saeLambda,beta,rho,transfer_function = tf.nn.sigmoid,
                                   optimizer = tf.train.AdamOptimizer(saeLearningRate))
    start1 = time.time()
    endEpoch = saeIterNum
    for i in range(saeIterNum):
        cost = encoderOne.partial_fit(trainImages)
        if cost < saeStop:
            endEpoch = i
            break
    print("One SAE Epoch:", '%04d' % (endEpoch), "cost=", "{:.9f}".format(cost))
    W1, W1_decode = encoderOne.getWeights()
    b1, b1_decode = encoderOne.getBiases()
    end1 = time.time()
    trainFeaturesOne = encoderOne.transform(trainImages)
    # 第二层隐层
    encoderTwo = SparseAutoencoder(hiddenSizeOne,hiddenSizeTwo,saeLambda,beta,rho,transfer_function = tf.nn.sigmoid,
                                   optimizer = tf.train.AdamOptimizer(saeLearningRate))
    start2 = time.time()
    endEpoch = saeIterNum
    for i in range(saeIterNum):
        cost = encoderTwo.partial_fit(trainFeaturesOne)
        if cost < saeStop:
            endEpoch = i
            break
    print("Two SAE Epoch:", '%04d' % (endEpoch), "cost=", "{:.9f}".format(cost))
    W2, W2_decode = encoderTwo.getWeights()
    b2, b2_decode = encoderTwo.getBiases()
    end2 = time.time()
    trainFeaturesTwo = encoderTwo.transform(trainFeaturesOne)
    # softmax层
    softTime, W_soft, b_soft = trainSoftmax(hiddenSizeTwo,numClass,softLearningRate,softIterNum,\
                                            trainFeaturesTwo,trainLabels,weight=softLambda,show=show)
    # 进行微调并且输出预测值和概率值
    weightPara = [W1, b1, W2, b2, W_soft, b_soft]
    fineTuneTime,trainProb,testPred,testProb,restPred,restProb = \
    fineTune(trainImages,trainLabels,testImages,testLabels,restImages,\
             weightPara, fineTuneLearningRate,fineTuneIterNum)
    number = trainImages.shape[0]
    selectNumber = int(0.66*number)
    for i in range(2):
        state = np.random.get_state()
        np.random.shuffle(trainImages)
        np.random.set_state(state)
        np.random.shuffle(trainLabels)	
        trainImages = trainImages[0:selectNumber,:]
        trainLabels = trainLabels[0:selectNumber,:]
        trainFeaturesOne = encoderOne.transform(trainImages)
        trainFeaturesTwo = encoderTwo.transform(trainFeaturesOne)
        softTime1, W_soft, b_soft = trainSoftmax(hiddenSizeTwo,numClass,softLearningRate,softIterNum,\
                                                trainFeaturesTwo,trainLabels,weight=softLambda,show=show)
        #进行微调并且输出预测值和概率值
        fineTuneTime1,_,_,_,_,restProb = \
        fineTune(trainImages,trainLabels,testImages,testLabels,restImages,\
                 W1,b1,W2,b2,W_soft,b_soft,fineTuneLearningRate,fineTuneIterNum)
        if i == 0:
            restProb2 = restProb
            time2 = softTime1 + fineTuneTime1
        else:
            restProb3 = restProb
            time3 = softTime1 + fineTuneTime1
    encoderOne.sessClose()
    encoderTwo.sessClose()
    tf.reset_default_graph()
    encodeOneTime = end1 - start1
    encodeTwoTime = end2 - start2
    runTime = encodeOneTime + encodeTwoTime + softTime + fineTuneTime + time2 + time3
    print('Running time: %s Seconds'%runTime)
    return trainProb, testPred, testProb, restPred, restProb, restProb2, restProb3, runTime

