import numpy as np
import math
import re
import os

trainDataSet = []
trainDescriptorSet = []
trainDataLabel = []
testDataSet = []
testDescriptorSet = []
testDataLabel = []
trainDataPosition = 0
testDataPosition = 0

def getNextBatch(isTrain=True,usingBatch=True,Batch=200):
    global trainDataPosition
    global testDataPosition
    if not usingBatch:
        if isTrain:
            return trainDataSet[:],trainDataLabel[:],trainDescriptorSet[:]
        else:
            return testDataSet[:],testDataLabel[:],testDescriptorSet[:]
    max = len(trainDataSet) if isTrain else len(testDataSet)
    if isTrain:
        next = max if trainDataPosition + Batch > max else trainDataPosition + Batch
        xs = trainDataSet[trainDataPosition:next]
        ys = trainDataLabel[trainDataPosition:next]
        ds = trainDescriptorSet[trainDataPosition:next]
        trainDataPosition = (trainDataPosition + Batch) % max
        while len(xs) < Batch:
            xs.append(trainDataSet[trainDataPosition])
            ys.append(trainDataLabel[trainDataPosition])
            ds.append(trainDescriptorSet[trainDataPosition])
            trainDataPosition += 1
    else:
        next = max if testDataPosition + Batch > max else testDataPosition + Batch
        xs = testDataSet[testDataPosition:next]
        ys = testDataLabel[testDataPosition:next]
        ds = testDescriptorSet[testDataPosition:next]
        testDataPosition = (testDataPosition + Batch) % max
        while len(xs) < Batch:
            xs.append(testDataSet[testDataPosition])
            ys.append(testDataLabel[testDataPosition])
            ds.append(testDescriptorSet[testDataPosition])
            testDataPosition += 1
    return xs,ys,ds




def cmp(a,b):
    for i in range(len(a)):
        if(a[i]!=b[i]):
            return False
    return True

def randomData(DataSet,Descriptor,Label):
    state = np.random.get_state()
    np.random.shuffle(DataSet)
    np.random.set_state(state)
    np.random.shuffle(Descriptor)
    np.random.set_state(state)
    np.random.shuffle(Label)

def std(descriptor):
    for j in len(descriptor[0]):
        max = -999999999.0
        min = 999999999.0
        for i in len(descriptor):
            if descriptor[i][j] > max:
                max = descriptor[i][j]
            if descriptor[i][j] < min:
                min = descriptor[i][j]
        d = max + 1 if max == min else max - min
        for i in len(descriptor):
            descriptor[i][j] = (descriptor[i][j] - min)/d

def readDataFile(path,list):
    file = open(path,"r")
    for line in file:
        nowline = line.strip().split(',')
        temp = []
        for i in range(1,len(nowline)):
            if nowline[i] == '0':
                temp.append(0.0)
            elif nowline[i] == '1':
                temp.append(1.0)
            else:
                try:
                    nowfloat = float(nowline[i])
                except:
                    nowfloat = 0.0
                if math.isnan(nowfloat):
                    nowfloat = 0.0
                temp.append(nowfloat)
        list.append(temp)

def getDataSet(projectDir):
    trainPosition = 0
    testPosition = 0
    datas = [[],[],[],[],[],[],[],[]]
    """
    nontox:1    train:2    descriptor:4 
    tox:   0    test :0    fpname    :0
    """
    allfile = os.listdir(projectDir)
    for file in allfile:
        nowfile = os.path.join(projectDir,file)
        index = 0
        if re.search("nontox",file):
            index += 1
        if re.search("train",file):
            index += 2
        if re.search("Descriptor",file):
            index += 4
        readDataFile(nowfile,datas[index])
        #####请好好的把这八个文件分类！！
        #是fpname
        if not index & 4 == 0:
            hastox = 1 - (index & 1)
            #是test
            if not index & 2 == 0:
                testDataSet.extend(datas[index])
                for i in range(len(datas[index])):
                    testDataLabel.append([hastox,1-hastox])
            #是train
            else:
                trainDataSet.extent(datas[index])
                for i in range(len(datas[index])):
                    trainDataLabel.append([hastox,1-hastox])
        #是descriptor
        else:
            #是test
            if not index & 2 == 0:
                testDescriptorSet.extend(datas[index])
            #是train
            else:
                trainDescriptorSet.extend(datas[index])


    randomData(trainDataSet,trainDescriptorSet,trainDataLabel)
    randomData(testDataSet,testDescriptorSet,testDataLabel)
    std(trainDescriptorSet)
    std(testDescriptorSet)

def main():
    return 0

if __name__ == '__main__':
    main()
