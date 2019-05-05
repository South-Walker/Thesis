import tensorflow as tf
import GetDataSet
import Inference

result = []
projectDir = r'C:\Users\lenovo\Desktop\毕业论文\result\des\projects\project0-0\FP' 
GetDataSet.getDataSet(projectDir)

LearningRateBase = 0.8
LearningRateDecay = 0.96

def train():
    x1 = tf.placeholder(tf.float32,[None,Inference.input1Node],name='x1-input')
    x2 = tf.placeholder(tf.float32,[None,Inference.input2Node],name='x2-input')
    label = tf.placeholder(tf.float32,[None,Inference.outputNode],name='label-input')
    y = Inference.inference(x1,x2)
    global_step = tf.Variable(0,False)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(label,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LearningRateBase,global_step,30,LearningRateDecay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    auc_value,auc_op = tf.metrics.auc(tf.argmax(label,1),tf.argmax(y,1))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        for i in range(5000):
            x1s,labels,x2s = GetDataSet.getNextBatch()
            sess.run(train_step,
                     feed_dict={x1:x1s,x2:x2s,label:labels})
            if i % 200 == 0:
                x1s,labels,x2s = GetDataSet.getNextBatch()
                a,b = sess.run([accuracy,loss],feed_dict={x1:x1s,x2:x2s,label:labels})
                print("train:%g\ntrainloss:%g" %(a,b))
                x1s,labels,x2s = GetDataSet.getNextBatch(False,False)
                a,b,c = sess.run([accuracy,loss,auc_op],feed_dict={x1:x1s,x2:x2s,label:labels})
                print("test:%g\ntestloss:%g" %(a,b))
                print("auc:%g" %(sess.run(auc_value,feed_dict={x1:x1s,x2:x2s,label:labels})))
            if i > 4000 and i  % 10 == 0:
                x1s,labels,x2s = GetDataSet.getNextBatch(False,False)
                result.append(sess.run(accuracy,feed_dict={x1:x1s,x2:x2s,label:labels}))
            
    with open(r'C:\Users\lenovo\Desktop\毕业论文\result\output.txt',"w") as f:
        for i in range(len(result)):
            f.write(str(result[i]))
            f.write("\n")


train()