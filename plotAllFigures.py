# -*- coding: utf-8 -*-
"""
Created on 2019.01.03

@author: Unclered
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def computeUseNumber(useNumber):
	# 对于后两类选择方式，计算出使用图片的数量
	useNumberList = list(useNumber)
	allUseNumber = []
	use = 0
	for i in useNumberList:
		use = use + i
		allUseNumber.append(use)
	return allUseNumber


def listAdd(listA, listB, listC=[]):#列表加法
	a = len(listA)
	b = len(listB)
	c = len(listC)
	if a != b:
		print('The length of two list is not equal')
		return 0
	else:
		if c == 0:
			listC = [0 for i in range(a)]
		d = [listA[i] + listB[i] + listC[i] for i in range(a)]
		return d


def listMinus(listA,a):  # 一个数减一个列表
	result = [a - i for i in listA]
	return result


def listMultiply(listA, b):  # 列表除以一个数字
	a = len(listA)
	result = [listA[i]*b for i in range(a)]
	return result


def computeResults(cycleNum,plotJudge,dataBase,method,time,name):
	# 设置参数
	numStep, numTest, numAll, numStart, numIncre, costWeight, timeWeight, dataWeight = initParameter(dataBase)
	figSize, lineW, msW, labelSize, fontSize = initSize()  # 设置图片和线条的大小
	decisionStep = list(range(1,numStep+1))  # 决策步骤，当然了后续也可以将这个变为调用的图片数
	dataSum = np.zeros([numStep, 14])
	for cycle in range(1, cycleNum + 1):
		data_path = 'Results/' + method + '/' + dataBase + time + str(cycle) + '.xlsx'
		data = pd.read_excel(data_path)
		# 获取各个需要求均值的数据
		dataSum += data.values
	average = dataSum / cycleNum
	twoCost = average[:, 0]     # 第1列是二支决策的误分类代价
	twoAcc = average[:, 1]      # 第2列是二支决策的分类精度
	twoAccIm = average[:, 2]    # 第3列是二支决策的impostor分类精度
	twoNgi = average[:, 3]      # 第4列是二支决策gallery误分到impostor的数量
	twoNig = average[:, 4]      # 第5列是二支决策impostor误分到gallery的数量
	threeCost = average[:, 5]   # 第6列是三支决策的误分类代价
	threeAcc = average[:, 6]    # 第7列是三支决策的分类精度
	threeAccIm = average[:, 7]  # 第8列是三支决策的impostor分类精度
	threeNgi = average[:, 8]    # 第9列是三支决策gallery误分到impostor的数量
	threeNig = average[:, 9]    # 第10列是三支决策impostor误分到gallery的数量
	threeNgb = average[:, 10]   # 第11列是三支决策gallery误分到boundary的数量
	threeNib = average[:, 11]   # 第12列是三支决策impostor误分到boundary的数量
	spendTime = average[:, 12]  # 第13列是该轮训练的所有时间，用于计算时间代价
	useNumber = average[:, 13]  # 第14列是本轮从剩余集抽取出的张数，用于计算获取数据代价
	positive = [numTest + i for i in threeNgb]
	negative = [numTest - i for i in threeNib]
	# dataUsed = [numStart + numIncre * i for i in range(numStep)]
	# 计算各个代价数据
	twoCostW = listMultiply(twoCost, costWeight)
	threeCostW = listMultiply(threeCost, costWeight)
	timeCost = listMultiply(spendTime, timeWeight)
	dataCost = listMultiply(useNumber, dataWeight)
	# dataCost = listMultiply(dataUsed, dataWeight)
	allTwoCost = listAdd(twoCostW, timeCost, dataCost)
	allThreeCost = listAdd(threeCostW, timeCost, dataCost)
	if plotJudge:
		# 开始制图
		name1 = dataBase + '_' + name + '_1'
		name2 = dataBase + '_' + name + '_2'
		name3 = dataBase + '_' + name + '_3'
		name4 = dataBase + '_' + name + '_4'
		
		# 图1：二支决策与三支决策的误分类代价对比
		plt.figure(name1,figsize=figSize)
		plt.plot(decisionStep, twoCost, 'k', linewidth=lineW, label='Two-way Decision', linestyle='--')
		plt.plot(decisionStep, threeCost, 'g', linewidth=lineW, label='Three-way Decision')
		# 设置图例和标记
		plt.xlabel('Decision Step', fontsize=fontSize)
		plt.ylabel('Decision Cost', fontsize=fontSize)
		plt.legend(prop={'size':labelSize})  # 显示图例
		plt.savefig('Figures/' + name1)

		# 图2：二支决策和三支决策的总代价
		plt.figure(name2, figsize=figSize)
		plt.plot(decisionStep, allTwoCost, 'k', linewidth=lineW, label='Two-way Decision', linestyle='--')
		twoIndex = allTwoCost.index(min(allTwoCost))
		twoMin = min(allTwoCost)
		plt.plot(twoIndex+1, twoMin, 'r', marker='*', ms=2*msW)
		plt.plot(decisionStep, allThreeCost, 'g', linewidth=lineW, label='Three-way Decision')
		threeIndex = allThreeCost.index(min(allThreeCost))
		threeMin = min(allThreeCost)
		plt.plot(threeIndex+1, threeMin, 'r', label='Minimum', marker='*', ms=2*msW)
		plt.xlabel('Decision Step', fontsize=fontSize)
		plt.ylabel('Total Cost', fontsize=fontSize)
		plt.legend(prop={'size':labelSize})#显示图例
		plt.savefig('Figures/' + name2)

		# 图3：二支决策与三支决策的分类精度对比
		plt.figure(name3,figsize=figSize)
		plt.plot(decisionStep, listMinus(twoAcc,1), 'k', linewidth=lineW, label='Two-way Decision', linestyle='--')
		plt.plot(decisionStep, listMinus(threeAcc,1), 'g', linewidth=lineW, label='Three-way Decision')
		plt.xlabel('Decision Step', fontsize=fontSize)
		plt.ylabel('Error rate', fontsize=fontSize)
		plt.legend(prop={'size': labelSize})  # 显示图例
		plt.savefig('Figures/' + name3)

		# 图4：三支决策的三个域
		plt.figure(name4, figsize=figSize)
		decisionStep.insert(0, 0)
		positive.insert(0, numAll)
		negative.insert(0, 0)
		plt.xlim(0, 25)
		if dataBase == 'EYaleB':
			# plt.ylim(numTest*2/3, numAll*2/3)#EYaleB
			plt.ylim(250, 500)
		elif dataBase  == 'PIE':
			# plt.ylim(0, numAll)# PIE
			plt.ylim(50, 350)
		plt.plot(decisionStep, positive, 'k', linewidth=lineW, marker='*',ms=msW)
		plt.plot(decisionStep, negative, 'g', linewidth=lineW, marker='o',ms=msW)
		upper = [numAll for i in decisionStep]
		under = [0 for i in decisionStep]
		plt.fill_between(decisionStep, positive, upper, color='red', alpha=0.5, label='positive region')
		plt.fill_between(decisionStep, positive, negative, color='yellow', label='boundary region')
		plt.fill_between(decisionStep, under, negative, color='blue', alpha=0.5, label='negative region')
		plt.xlabel('Decision Step', fontsize=fontSize)
		plt.ylabel('Three Regions', fontsize=fontSize)
		plt.legend(prop={'size':labelSize})  # 显示图例
		plt.savefig('Figures/' + name4)
		# plt.show()#绘制图

	return useNumber, twoCost, threeCost, allTwoCost, allThreeCost, threeAcc, threeAccIm


def initParameter(dataBase):
	if dataBase == 'EYaleB':
		numStep = 25
		numTest = 24 * 15
		numAll = 24 * 35
		numStart = 175
		numIncre = 25
		costWeight = 1
		timeWeight = 4
		dataWeight = 0.7
	elif dataBase == 'PIE':
		numStep = 25
		numTest = 12 * 15
		numAll = 12 * 35
		numStart = 70
		numIncre = 8
		costWeight = 1
		timeWeight = 10
		dataWeight = 2.5

	return numStep, numTest, numAll, numStart, numIncre, costWeight, timeWeight, dataWeight


def initSize():
	figSize = (6.5, 7)
	lineW = 5
	msW = 10
	labelSize = 16
	fontSize = 20
	return figSize, lineW, msW, labelSize, fontSize


#  dataBase = dataBaseList[0]中间的0要改为1一次
time1 = '_0129_'
time2 = '_0129_'
dataBaseList = ['EYaleB', 'PIE', 'AR']
dataBase = dataBaseList[0]
numStep, numTest, numAll, numStart, numIncre, costWeight, timeWeight, dataWeight, = initParameter(dataBase)
methodList = ['randomSelect', 'CSmarginSample']
name1 = 'RandomSample'
name2 = 'CSminMargin'
cycleNum = 10
onePlotJudge = 1
twoPlotJudge = 1
comparePlotJudge = 1
lineW = 2
useNumber1, twoCost1, threeCost1, allTwoCost1, allThreeCost1, threeAcc1, threeAccIm1 = \
computeResults(cycleNum, onePlotJudge, dataBase, methodList[0], time2, name1)
useNumber2, twoCost2, threeCost2, allTwoCost2, allThreeCost2, threeAcc2, threeAccIm2 = \
computeResults(cycleNum, twoPlotJudge, dataBase, methodList[1], time1, name2)
if comparePlotJudge:
	useNumber1 = list(range(1, numStep + 1))
	useNumber2 = list(range(1, numStep + 1))
	xName = 'Decision Step'
	# 绘制对比图
	fig = plt.figure('Comparison on ' + dataBase, figsize=(18,3))
	fig.suptitle('Comparison of ' + name1 + ' and ' + name2 + ' on ' + dataBase, fontsize=15)

	# 图1
	ax1 = fig.add_subplot(131)
	ax1.plot(useNumber1, threeCost1, 'k', linewidth=lineW, label=name1, linestyle='--')
	ax1.plot(useNumber2, threeCost2, 'g', linewidth=lineW, label=name2)
	ax1.set_xlabel(xName, fontsize=12)
	ax1.set_ylabel('Decision Cost', fontsize=12)
	ax1.legend()  # 显示图例

	# 图2
	ax2 = fig.add_subplot(132)
	ax2.plot(useNumber1, allThreeCost1, 'k', linewidth=lineW, label=name1, linestyle='--')
	ax2.plot(useNumber2, allThreeCost2, 'g', linewidth=lineW, label=name2)
	ax2.set_xlabel(xName, fontsize=12)
	ax2.set_ylabel('Total Cost', fontsize=12)
	if dataBase == 'EYaleB':
		ax2.set_ylim(600,1200)  # EYaleB
	elif dataBase == 'PIE':
		ax2.set_ylim(850,1300)  # PIE
	ax2.legend()  # 显示图例

	# 图3
	ax3 = fig.add_subplot(133)
	ax3.plot(useNumber1, listMinus(threeAcc1,1), 'k', linewidth=lineW, label=name1, linestyle='--')
	ax3.plot(useNumber2, listMinus(threeAcc2,1), 'g', linewidth=lineW, label=name2)
	ax3.set_xlabel(xName, fontsize=12)
	ax3.set_ylabel('Error rate', fontsize=12)
	ax3.legend()  # 显示图例
	plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85, wspace=0.5)
	plt.savefig('Figures/' + dataBase)
	# plt.show()#绘制图