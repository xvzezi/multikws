'''
Module : Neural Netword Model Getter
-----------------------
Features :
1. 获取训练所需的神经网络结构
2. 结构有tensorflow，tflearn两种

-----------------------
Version :
2018.06.06  第一版

'''

############ 前置module ################
import tensorflow as tf 
import tflearn
#-------------------------------------#
import tf.contrib.layers as tflayers 
#######################################

class DNN:

    @classmethod
    def TensorflowDNN(cls, layer_num, neural_num, vector_dim, class_num, batch_size):
        inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, vector_dim])

        # ctc_loss 需要的矩阵
        targets = tf.sparse_placeholder(tf.int32)

        # 1维向量 
        seq_len = tf.placeholder(tf.int32, [None])

        # 定义DNN网络
        hidden = tflayers.fully_connected(inputs, neural_num, scope='hidden1')
        for i in range(layer_num):
            hidden = tflayers.fully_connected(hidden, neural_num, scope='hidden%d' % i)
        logits = tflayers.fully_connected(hidden, num_outputs=class_num, scope='output', activation_fn=None)

        # 转成时序为主
        logits = tf.transpose(logits, (1, 0, 2))

        return logits, inputs, targets, seq_len 
    
    @classmethod
    def TFLearnDNN(cls, layer_num, neural_num, vector_dim, class_num, batch_size):
        inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, vector_dim])

        # ctc_loss 需要的矩阵
        targets = tf.sparse_placeholder(tf.int32)

        # 1维向量 
        seq_len = tf.placeholder(tf.int32, [None])

        # 定义网络
        net = tflearn.fully_connected(inputs, neural_num, activation='relu') 
        for i in range(layer_num):
            net = tflearn.fully_connected(net, neural_num, activation='relu')
        net = tflearn.fully_connected(net, class_num)

        net = tf.transpose(net, (1, 0, 2))
        return net, inputs, targets, seq_len 

