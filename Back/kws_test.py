'''
Module : DNN Tester
-----------------------
Features :
1. 测试目标文件

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
from Tool.sparse_tensor import sparse_tuple_from
from Tool.sparse_tensor import report_accuracy
from Tool.sparse_tensor import decode_sparse_tensor
#######################################

def TestTargetFileAsWhole(test_path, model_path, test_round, ltype='PH', ltone=False):
    # prepare the corpus
    logger = KWSError('./log/whole_test.log')
    logger_head = logger.acquireHeader('kws_test->Whole')
    corpus = kws_data.DataBase(kws_log=logger)
    corpus.LoadMeta(test_path)
    corpus.LabelSetting(ltype, ltone)
    corpus.AudioSetting(fcmvn=False)
    decode_dict = corpus.GetDecodeDict()
    feature_dim,class_num = corpus.GetDimensionInfo()

    # prepare the network 
    logits, inputs, targets, seq_len = kws_nn.DNN.TensorflowDNN(layer_num=3,
                                                                neural_num=256,
                                                                vector_dim=feature_dim,
                                                                class_num=class_num+1,
                                                                batch_size=1)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    distance_measure = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    init = tf.global_variables_initializer()

    # prepare the model 
    check_point = tf.train.get_checkpoint_state(model_path)

    # start the session 
    with tf.Session() as session:
        session.run(init)
        model_reader = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        model_reader.restore(session, check_point.model_checkpoint_path)

        for i in range(test_round):
            cur_inputs, cur_targets, cur_seq_len = corpus.GetNextBatch(1)
            cur_targets = sparse_tuple_from(cur_targets)
            test_feed = {
                inputs:     cur_inputs,
                targets:    cur_targets,
                seq_len:    cur_seq_len
            }
            recognized, probs, measure = session.run([
                decoded, log_prob, distance_measure 
            ], test_feed)
            logger.print(logger_head, report_accuracy(recognized[0], cur_targets, measure, decode_dict))
    
    logger.record()


def cutWholeIntoFrame(audio_origin):
    cutted = []
    for audio in audio_origin:
        # 该循环对 第i个 音频切分
        for frame in audio:
            # 拿出每一帧，每一帧当作[batch_size=1, time_step=1, feature_dim]的向量
            cutted.append([[frame]])
    return cutted

def removeRepeated(origin):
    res = []
    cnt = []
    cur_char = origin[0]
    cur_cnt = 0
    res.append(cur_char)
    for i in origin:
        if i != cur_char:
            cnt.append(cur_cnt)
            res.append(i)
            cur_char = i
            cur_cnt = 1 
        else:
            cur_cnt += 1 
    cnt.append(cur_cnt)
    return res, cnt 

def TestTargetFileAsFrame(test_path, model_path, test_round, ltype='PH', ltone=False):
        # prepare the corpus
    logger = KWSError('./log/frame_test.log')
    logger_head = logger.acquireHeader('kws_test->Whole')
    corpus = kws_data.DataBase(kws_log=logger)
    corpus.LoadMeta(test_path)
    corpus.LabelSetting(ltype, ltone)
    corpus.AudioSetting('''fcmvn=False''')
    decode_dict = corpus.GetDecodeDict()
    feature_dim,class_num = corpus.GetDimensionInfo()

    # prepare the network 
    logits, inputs, targets, seq_len = kws_nn.DNN.TensorflowDNN(layer_num=3,
                                                                neural_num=256,
                                                                vector_dim=feature_dim,
                                                                class_num=class_num+1,
                                                                batch_size=1)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # distance_measure = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    init = tf.global_variables_initializer()

    # prepare the model 
    check_point = tf.train.get_checkpoint_state(model_path)

    # start the session 
    with tf.Session() as session:
        session.run(init)
        model_reader = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        model_reader.restore(session, check_point.model_checkpoint_path)

        for i in range(test_round):
            start = time.time()
            cur_inputs, cur_targets, cur_seq_len = corpus.GetNextBatch(1)
            audio_frames = cutWholeIntoFrame(cur_inputs)
            rec_res = []
            for frame in audio_frames:
                test_feed = {
                    inputs:     frame,
                    seq_len:    [1]
                }
                recognized, probs = session.run([
                    decoded, log_prob
                ], test_feed)
                recognized = decode_sparse_tensor(recognized[0], decode_dict)
                rec_res.append(recognized[0])
            avrg = start / len(audio_frames)
            dd, cnt = removeRepeated(rec_res)
            start = time.time() - start
            logger.print(logger_head, 'time:{}, average:{}, len:{}'.format(start, avrg, len(audio_frames)))
            logger.print(logger_head, '{}'.format(rec_res))
            logger.print(logger_head, '{}'.format(dd))
            logger.print(logger_head, '{}'.format(cnt))
            logger.print(logger_head, '{}'.format(cur_targets))
    
    logger.record()


