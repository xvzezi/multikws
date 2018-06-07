'''
Module : 将训练文件分类
-----------------------
Features :
1. 输出错误信息

-----------------------
Version :
2018.06.07  第一版

'''

############ 前置module ################
import os 
import shutil
#-------------------------------------#

#######################################

def pickUpFromCorpus(corpus_path, target_path, id):
    if not os.path.exists(target_path):
            os.makedirs(target_path)                
    amount = 0
    # walk through the files
    for (dir_path, _, file_names) in os.walk(corpus_path):
        for filename in file_names:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                core_name = filename.split('.')[0]
                names = core_name.split('_')
                if names[1] == str(id):
                    shutil.copy(os.sep.join([dir_path, filename]), target_path)
                    print('copying %s -> %s' % (filename, target_path))
                    amount += 1
    print('Total amount of Files copied: %d' % amount)
    return amount 


if __name__ == "__main__":
    corpus = 'F:/ASR/THCHS30/data_thchs30/train'
    dst = 'F:/ASR/THCHS30/data_thchs30/0train'
    pickUpFromCorpus(corpus, dst, 0)