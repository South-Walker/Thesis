import numpy as np
import math
import re
import os

trainName = []
testName = []

trainDataSet = []
trainDescriptorSet = []
trainDataLabel = []
testDataSet = []
testDescriptorSet = []
testDataLabel = []
trainDataPosition = 0
testDataPosition = 0

def getNextBatch(isTrain=True,usingBatch=True,Batch=100):
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

def randomData(DataSet,Descriptor,Label,Name):
    state = np.random.get_state()
    np.random.shuffle(DataSet)
    np.random.set_state(state)
    np.random.shuffle(Descriptor)
    np.random.set_state(state)
    np.random.shuffle(Label)
    np.random.set_state(state)
    np.random.shuffle(Name)

def std(descriptor):
    for j in range(len(descriptor[0])):
        max = -999999999.0
        min = 999999999.0
        for i in range(len(descriptor)):
            if descriptor[i][j] > max:
                max = descriptor[i][j]
            if descriptor[i][j] < min:
                min = descriptor[i][j]
        d = max - min
        for i in range(len(descriptor)):
            descriptor[i][j] = (descriptor[i][j] - min)/d if d != 0 else 0.00001
            if descriptor[i][j] != descriptor[i][j]:
                descriptor[i][j] = 0.00001

def readDataFile(path,list,namelist):
    file = open(path,"r")
    for line in file:
        nowline = line.strip().split(',')
        temp = []
        #第一个是名字
        namelist.append(nowline[0])
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
                temp.append(nowfloat)
        list.append(temp)

def getDataSet(projectDir):
    trainPosition = 0
    testPosition = 0
    datas = [[],[],[],[],[],[],[],[]]
    names = [[],[],[],[],[],[],[],[]]
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
        readDataFile(nowfile,datas[index],names[index])
        #####请好好的把这八个文件分类！！
    for index in range(len(datas)):
        #是fpname
        if index & 4 == 0:
            hastox = 1 - (index & 1)
            #是test
            if index & 2 == 0:
                testDataSet.extend(datas[index])
                for i in range(len(datas[index])):
                    testDataLabel.append([hastox,1-hastox])
                testName.extend(names[index])
            #是train
            else:
                trainDataSet.extend(datas[index])
                for i in range(len(datas[index])):
                    trainDataLabel.append([hastox,1-hastox])
                trainName.extend(names[index])
        #是descriptor
        else:
            #是test
            if index & 2 == 0:
                testDescriptorSet.extend(datas[index])
            #是train
            else:
                trainDescriptorSet.extend(datas[index])
    randomData(trainDataSet,trainDescriptorSet,trainDataLabel,trainName)
    randomData(testDataSet,testDescriptorSet,testDataLabel,testName)
    std(trainDescriptorSet)
    std(testDescriptorSet)

def save(list,path):
    file = open(path,"w")
    for i in range(len(list)):
        for j in range(len(list[i])):
            file.write(str(list[i][j]))
            if j != len(list[i])-1:
                file.write(',')
        file.write('\n')
    
def main():
    return 0

if __name__ == '__main__':
    main()
