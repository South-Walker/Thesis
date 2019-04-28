import tensorflow as tf
import GetDataSet

result = []
projectDir = r'C:\Users\lenovo\Desktop\毕业论文\result\fps\projects\project0-0\MACCSFP' 
GetDataSet.getDataSet(projectDir)

inputNode = len(GetDataSet.trainDataSet[0])
outputNode = 2
layer1Node = 100

global_step = tf.Variable(0,False)
keeprate = tf.placeholder(tf.float32)
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.94,staircase=True)
x = tf.placeholder(tf.float32,(None,inputNode),name="x-input")
label = tf.placeholder(tf.float32,(None,outputNode),name="label-input")
w1 = tf.Variable(tf.truncated_normal([inputNode,layer1Node],mean=0.0,
                                     stddev=0.1,dtype=tf.float32,seed=1996))
biases1 = tf.Variable(tf.constant(0.1,tf.float32,[layer1Node]))
w2 = tf.Variable(tf.truncated_normal([layer1Node,outputNode],mean=0.0,
                                     stddev=0.1,dtype=tf.float32,seed=1996))
biases2 = tf.Variable(tf.constant(0.1,tf.float32,[outputNode]))

layer1 = tf.nn.relu(tf.matmul(x,w1)
                    + biases1)
layer1_drop = tf.nn.dropout(layer1,keeprate)
y = tf.nn.relu(tf.matmul(layer1_drop,w2)
                    + biases2)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=y,labels=tf.argmax(label,1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
regularizer = tf.contrib.layers.l2_regularizer(0.003)
regularization = regularizer(w1) + regularizer(w2)


loss = cross_entropy_mean + regularization
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
train_step = tf.train.AdamOptimizer().minimize(loss,global_step)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(5000):
        xs,labels = GetDataSet.getNextBatch()
        sess.run(train_step,
                 feed_dict={x:xs,label:labels,keeprate:1})
        if i % 200 == 0:
            xs,labels = GetDataSet.getNextBatch(True,False)
            a,b = sess.run([accuracy,loss],feed_dict={x:xs,label:labels,keeprate:1})
            print("train:%g\ntrainloss:%g" %(a,b))
            xs,labels = GetDataSet.getNextBatch(False,False)
            a,b = sess.run([accuracy,loss],feed_dict={x:xs,label:labels,keeprate:1})
            print("test:%g\ntestloss:%g" %(a,b))
        if i > 4000 and i  % 10 == 0:
            xs,labels = GetDataSet.getNextBatch(False,False)
            result.append(sess.run(accuracy,feed_dict={x:xs,label:labels,keeprate:1}))
            
with open(r'C:\Users\lenovo\Desktop\毕业论文\result\output.txt',"w") as f:
    for i in range(len(result)):
        f.write(str(result[i]))
        f.write("\n")