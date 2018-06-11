'''
Module : DNN Trainer
-----------------------
Features :
1. 训练
2. 结构有tensorflow，tflearn两种

-----------------------
Version :
2018.06.06  第一版

'''

############ 前置module ################
import tensorflow as tf 
import tflearn
import numpy as np 
import time 
#-------------------------------------#
import kws_nn
import kws_data
from Tool.kws_error import KWSError
#######################################

class Trainer:
    def __init__(self, corpus_path, save_path, mode='dev'):
        self.dev_path = corpus_path + '/dev'
        self.train_path = corpus_path + '/train'
        self.save_path = save_path
        self.corpus_path = ''
        if mode == 'dev':
            self.corpus_path = self.dev_path
        else:
            self.corpus_path = corpus_path + '/' + mode 
        self.logger = KWSError('./Log/train.log')
        self.log_head = self.logger.acquireHeader('kws_train->Trainer')

        self.cur_inputs = []
        self.cur_targets = []
        self.cur_seq_len = []       
        self.check_point = None  

        # prepare the corpus
        self.corpus = kws_data.DataBase(kws_log=self.logger)
        self.corpus.LoadMeta(self.corpus_path)
        
    
    def log(self, content):
        self.logger.print(self.log_head, content)

    def log_end(self):
        self.logger.record()
    
    ################### Train #######################
    ## Trainer based on tensorflow module
    def train(self, learning_rate, decay_fator, ltype='PH', ltone=False, batch_size=32, n_epcho=10, resume=False):
        # prepare the corpus
        # self.corpus = kws_data.DataBase(kws_log=self.logger)
        # self.corpus.LoadMeta(self.corpus_path)
        self.corpus.LabelSetting(ltype, ltone)
        self.corpus.AudioSetting('''fcmvn=False''')
        self.decode_dict = self.corpus.GetDecodeDict()

        # other Tool
        self.model_saver = None # saver
        self.merged = None      # summary      
        self.sum_writer = None  # summary writer
        if resume:
            self.check_point = tf.train.get_checkpoint_state(self.save_path)

        # prepare the training step 
        self.n_epcho = n_epcho
        self.batch_size = batch_size
        self.current_epcho = tf.Variable(0, trainable=False)
        self.n_batch = len(self.corpus.audio_files) // batch_size
        self.decay_steps = self.n_batch * n_epcho
        self.learning_rate = tf.train.exponential_decay(learning_rate, 
                                                    self.current_epcho, 
                                                    self.decay_steps,
                                                    decay_fator,
                                                    staircase=True)
        feature_dim, class_num = self.corpus.GetDimensionInfo()
        self.logits, self.inputs, self.targets, self.seq_len = kws_nn.DNN.TensorflowDNN(layer_num=3,
                                                                                        neural_num=256,
                                                                                        vector_dim=feature_dim,
                                                                                        class_num=class_num+1,
                                                                                        batch_size=batch_size)
        ## ctc loss
        self.loss = tf.nn.ctc_loss(labels=self.targets, inputs=self.logits, sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('ctc_loss', self.cost) 

        # optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=current_epcho)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=current_epcho)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.current_epcho)

        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len)
        self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))
        self.init = tf.global_variables_initializer()

        # go training
        with tf.Session() as session:
            session.run(self.init) 
            self.model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            if resume:
                self.model_saver.restore(session, self.check_point.model_checkpoint_path)
            self.merged = tf.summary.merge_all()
            self.sum_writer = tf.summary.FileWriter(logdir=self.save_path, graph=session.graph)
            global_step = 0
            total_time_start = time.time()
            for curr_epcho in range(n_epcho):
                self.log('Epcho {} :'.format(curr_epcho))
                start_time = time.time()
                train_cost = 0
                batch_time_acc = 0
                for curr_batch in range(self.n_batch):
                    batch_start_time = time.time()
                    batch_cost, batch_steps = self.do_tensorflow_batch(session)
                    global_step = batch_steps
                    train_cost += batch_cost * self.batch_size
                    batch_end_time = time.time()
                    batch_time_acc += batch_end_time - batch_start_time
                    if curr_batch % 5 == 0:
                        self.log('Step: {}, Last 5 Batch Cost Total: {}'.format(batch_steps, batch_time_acc))
                        batch_time_acc = 0
                # after an epcho 
                train_cost /= self.batch_size * self.n_batch    # average loss in the epcho
                epcho_time = time.time() - start_time
                self.do_report_on_epcho(session, curr_epcho, train_cost, epcho_time)
            # all the epcho done 
            self.log('Train finished. Total Time costed(s): {}'.format(time.time() - total_time_start))
            self.save_cur_model(session, global_step)
            self.sum_writer.close()



    ## Trainer Based on TFLearn Module
    ###################### Helper Func #################################
    def do_report_on_epcho(self, session, cur_ep, average_cost, time_consume):
        val_feed = {self.inputs: self.cur_inputs,
                     self.targets: self.cur_targets,
                     self.seq_len: self.cur_seq_len}
        val_cost, val_acc, lr, steps = session.run([
            self.cost, self.acc, self.learning_rate, self.current_epcho
        ], feed_dict=val_feed)
        self.log('Epcho {}/{}, steps {}, average train cost : {}'.format(
            cur_ep+1, self.n_epcho, steps, average_cost
        ))
        self.log('On Evaluation: cost {}, acc {}, learning rate {}, time {}'.format(
            val_cost, val_acc, lr, time_consume
        ))

    def do_report(self, session):
        test_feed = {self.inputs: self.cur_inputs,
                     self.targets: self.cur_targets,
                     self.seq_len: self.cur_seq_len}
        dd, log_probs, accuracy = session.run([self.decoded[0], self.log_prob, self.acc], test_feed)
        self.log(self.report_accuracy(dd, self.cur_targets, accuracy, self.decode_dict))

    def do_tensorflow_batch(self, session):
        self.cur_inputs, self.cur_targets, self.cur_seq_len = self.corpus.GetNextBatch(self.batch_size)
        # print(np.shape(self.cur_inputs))
        # exit()
        # 将targets转化为稀疏矩阵
        self.cur_targets = self.sparse_tuple_from(self.cur_targets)
        feed = {self.inputs: self.cur_inputs,
                self.targets: self.cur_targets,
                self.seq_len: self.cur_seq_len}
        b_merged, b_loss, _, _, _, b_cost, steps, _ = session.run(
            [self.merged, self.loss, self.targets, self.logits, self.seq_len, 
            self.cost, self.current_epcho, self.optimizer], feed_dict=feed)
        self.sum_writer.add_summary(b_merged, steps)
        if steps % 5 == 0:
            self.log('cost:{}, steps:{}'.format(b_cost, steps))
        if steps > 0 and steps % 100 == 0:
            self.do_report(session)
            if steps % 500 == 0:
                self.save_cur_model(session, steps)
        return b_cost, steps 

    def save_cur_model(self, session, steps):
        return self.model_saver.save(session, self.save_path + '/kws', global_step=steps)
    
    @classmethod
    def sparse_tuple_from(cls, sequences, dtype=np.int32):
        """
        Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    @classmethod
    def decode_a_seq(cls, indexes, spars_tensor, trans_dict=None):
        decoded = []
        for m in indexes:
            label = spars_tensor[1][m]
            if trans_dict is not None:
                label = trans_dict[label]
            decoded.append(label)
        return decoded


    @classmethod
    def decode_sparse_tensor(cls, sparse_tensor, trans_dict=None):
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)
        result = []
        for index in decoded_indexes:
            result.append(cls.decode_a_seq(index, sparse_tensor, trans_dict))
        return result

    @classmethod
    def report_accuracy(cls, decoded_list, test_targets, sofar_acc, decode_dict):
        original_list = cls.decode_sparse_tensor(test_targets, decode_dict)
        detected_list = cls.decode_sparse_tensor(decoded_list, decode_dict)
        true_number = 0

        if len(original_list) != len(detected_list):
            return 'Length to test not matched with decoded: %d : %d' % (len(original_list), len(detected_list))
        
        content = "T/F: original(length) <-------> detectcted(length) with so far acc {}\n".format(sofar_acc)
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            content += '{}: \n{} ({})\n{} ({})\n'.format(hit, number, len(number), detect_number, len(detect_number))
            if hit:
                true_number = true_number + 1
        content += "Test Accuracy: {}\n".format(true_number * 1.0 / len(original_list))
        return content 

        