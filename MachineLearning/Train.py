import tensorflow as tf
import GetDataSet
import Inference

def train():
    x1 = tf.placeholder(tf.float32,[None,Inference.input1Node],name='FP-input')
    x2 = tf.placeholder(tf.float32,[None,Inference.input2Node],name='Des-input')
    keeprate = tf.placeholder(tf.float32,name='keeprate')
    label = tf.placeholder(tf.float32,[None,Inference.outputNode],name='Label-input')
    one = tf.ones_like(label,name='ones')
    zero = tf.zeros_like(label,name='zeros')
    t = Inference.fakerinference(x1,x2,keeprate)
    y = tf.where(t < 0.5,zero,one)
    global_step = tf.Variable(0,False)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=t,labels=tf.argmax(label,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='MeanCrossEntropy')
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'),name='Regularizer')
    
    train_step = tf.train.AdamOptimizer().minimize(loss,global_step,name='TrainStep')
    #
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(label,1),name='CorrectNum')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='Acc')
    auc_value,auc_op = tf.metrics.auc(tf.argmax(label,1),tf.argmax(y,1),name="Auc")
    traindata,trainlabel,traindesc = GetDataSet.getNextBatch(True,False)
    testdata,testlabel,testdesc = GetDataSet.getNextBatch(False,False)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        for i in range(5000):

            x1s,labels,x2s = GetDataSet.getNextBatch()
            sess.run(train_step,
                     feed_dict={x1:x1s,x2:x2s,label:labels,keeprate:0.75})

            if i % 10 == 0:
                print(i)
                a,b = sess.run([accuracy,loss],
                               feed_dict={x1:traindata,x2:traindesc,label:trainlabel,keeprate:1})
                trainacc.append(a)
                trainloss.append(b)
                a,b,c = sess.run([accuracy,loss,auc_op],
                                 feed_dict={x1:testdata,x2:testdesc,label:testlabel,keeprate:1})
                testacc.append(a)
                testloss.append(b)
                auc.append(sess.run(auc_value,
                                    feed_dict={x1:testdata,x2:testdesc,label:testlabel,keeprate:1}))
        writer.close()
trainloss = []
trainacc = []
testloss = []
testacc = []
auc = []

projectDir = r'C:\Users\lenovo\Desktop\毕业论文\result\des\projects\project0-4\FP'
GetDataSet.getDataSet(projectDir)
train()
projectDir = r'C:\Users\lenovo\Desktop\毕业论文\result\des\projects'
GetDataSet.save(trainacc,projectDir+'\\trainacc.csv',1)
GetDataSet.save(trainloss,projectDir+'\\trainloss.csv',1)
GetDataSet.save(testacc,projectDir+'\\testacc.csv',1)
GetDataSet.save(testloss,projectDir+'\\testloss.csv',1)
GetDataSet.save(auc,projectDir+'\\auc.csv',1)