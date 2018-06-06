'''
Module : DNN Trainer
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
import kws_nn
import kws_data
if __name__ == '__main__':
    import sys 
    sys.path.append('../')
    from multikws.Tool.kws_error import KWSError
else:
    from Tool.kws_error import KWSError
#######################################

class Trainer:
    def __init__(self, corpus_path, mode='dev'):
        self.dev_path = corpus_path + '/dev'
        self.train_path = corpus_path + '/train'
        self.corpus_path = ''
        if mode == 'dev':
            self.corpus_path = self.dev_path
        else:
            self.corpus_path = self.train_path
        self.logger = KWSError('./Log/train.log')
        self.log_head = self.logger.acquireHeader('kws_train->Trainer')
        
    
    def log(self, content):
        self.logger.print(self.log_head, content)

    def train(self, learning_rate, decay_fator, ltype='PH', ltone=False, batch_size=32, n_epcho=10):
        # prepare the corpus
        corpus = kws_data.DataBase(kws_log=self.logger)
        corpus.LoadMeta(self.corpus_path)
        corpus.LabelSetting(ltype, ltone)
        corpus.AudioSetting()

        # prepare the training step 
        current_epcho = tf.Variable(0, trainable=False)
        n_batch = len(corpus.audio_files) // batch_size
        decay_steps = n_batch * n_epcho
        learning_rate = tf.train.exponential_decay(learning_rate, 
                                                    current_epcho, 
                                                    decay_steps,
                                                    decay_fator,
                                                    staircase=True)
        feature_dim, class_num = corpus.GetDimensionInfo()
        logits, inputs, targets, seq_len = kws_nn.DNN.TensorflowDNN(layer_num=3,
                                                                    neural_num=256,
                                                                    vector_dim=feature_dim,
                                                                    class_num=class_num+1,
                                                                    batch_size=batch_size)
        ## ctc loss
        loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
        cost = tf.reduce_mean(loss)
        tf.summary.scalar('ctc_loss', cost) 

        # optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=current_epcho)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=current_epcho)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=current_epcho)

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
        init = tf.global_variables_initializer()


