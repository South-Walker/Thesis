import tensorflow as tf

input1Node = 1024
input2Node = 729
outputNode = 2
layer1Node = 200
regularizerRate = 0.1
keepRate = 0.8

def getWeightVariable(shape):
    weights = tf.get_variable(
        "weights",shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizerRate != 0:
        tf.add_to_collection('losses',
                             tf.contrib.layers.l2_regularizer(regularizerRate)(weights))
    return weights

def inference(inputTensor1,inputTensor2):
    with tf.variable_scope('layer1'):
        weights = getWeightVariable([input1Node,layer1Node])
        biases = tf.get_variable("biases",[layer1Node],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.concat(
            [tf.nn.relu(tf.matmul(inputTensor1,weights) + biases) , inputTensor2],1)
    layer1_drop = tf.nn.dropout(layer1,keepRate)
    with tf.variable_scope('layer2'):
        weights = getWeightVariable([layer1Node + input2Node,outputNode])
        biases = tf.get_variable("biases",[outputNode],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1_drop,weights) + biases
    return layer2