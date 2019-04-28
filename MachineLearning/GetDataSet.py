import numpy as np
import re
import os

trainDataSet = []
trainDataLabel = []
testDataSet = []
testDataLabel = []
trainDataPosition = 0
testDataPosition = 0

def getNextBatch(isTrain=True,usingBatch=True,Batch=200):
    global trainDataPosition
    global testDataPosition
    if not usingBatch:
        if isTrain:
            return trainDataSet[:],trainDataLabel[:]
        else:
            return testDataSet[:],testDataLabel[:]
    max = len(trainDataSet) if isTrain else len(testDataSet)
    if isTrain:
        next = max if trainDataPosition + Batch > max else trainDataPosition + Batch
        xs = trainDataSet[trainDataPosition:next]
        ys = trainDataLabel[trainDataPosition:next]
        trainDataPosition = (trainDataPosition + Batch) % max
        while len(xs) < Batch:
            xs.append(trainDataSet[trainDataPosition])
            ys.append(trainDataLabel[trainDataPosition])
            trainDataPosition += 1
    else:
        next = max if testDataPosition + Batch > max else testDataPosition + Batch
        xs = testDataSet[testDataPosition:next]
        ys = testDataLabel[testDataPosition:next]
        testDataPosition = (testDataPosition + Batch) % max
        while len(xs) < Batch:
            xs.append(testDataSet[testDataPosition])
            ys.append(testDataLabel[testDataPosition])
            testDataPosition += 1
    return xs,ys




def cmp(a,b):
    for i in range(len(a)):
        if(a[i]!=b[i]):
            return False
    return True

def randomData(DataSet,Label):
    state = np.random.get_state()
    np.random.shuffle(DataSet)
    np.random.set_state(state)
    np.random.shuffle(Label)

def readDataFile(list,path):
    file = open(path,"r")
    count = 0;
    for line in file:
        nowline = line.strip().split(',')
        temp = []
        for i in range(1,len(nowline)):
            if nowline[i] == '0':
                temp.append(0)
            elif nowline[i] == '1':
                temp.append(1)
            else:
                raise RuntimeError()
        list.append(temp)
        count += 1
    return count

def fillDataSet(projectDir):
    allfile = os.listdir(projectDir)
    for file in allfile:
        nowfile = os.path.join(projectDir,file)
        if re.search("nontox",file):
            value = 0
        else:
            value = 1
        if re.search("train",file):
            count = readDataFile(trainDataSet,nowfile)
            for i in range(count):
                trainDataLabel.append([value,1-value])
        else:
            count = readDataFile(testDataSet,nowfile)
            for i in range(count):
                testDataLabel.append([value,1-value])

def getDataSet(projectDir):
    trainPosition = 0
    testPosition = 0
    fillDataSet(projectDir)
    randomData(trainDataSet,trainDataLabel)
    randomData(testDataSet,testDataLabel)

def main():
    projectDir = r'C:\Users\lenovo\Desktop\毕业论文\result\fps\projects\project0-1\FP'
    getDataSet(projectDir)

if __name__ == '__main__':
    main()
