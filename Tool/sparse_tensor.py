'''
Module : Sparse tensor convertor
-----------------------
Features :
1. 将dense tensor 与 sparse tensor 互相转化

-----------------------
Version :
2018.06.08  第一版

'''

############ 前置module ################
import numpy as np 
#-------------------------------------#

#######################################


def sparse_tuple_from(sequences, dtype=np.int32):
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


def decode_a_seq(indexes, spars_tensor, trans_dict=None):
    decoded = []
    for m in indexes:
        label = spars_tensor[1][m]
        if trans_dict is not None:
            label = trans_dict[label]
        decoded.append(label)
    return decoded



def decode_sparse_tensor(sparse_tensor, trans_dict=None):
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
        result.append(decode_a_seq(index, sparse_tensor, trans_dict))
    return result

def report_accuracy(decoded_list, test_targets, sofar_acc, decode_dict):
    original_list = decode_sparse_tensor(test_targets, decode_dict)
    detected_list = decode_sparse_tensor(decoded_list, decode_dict)
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