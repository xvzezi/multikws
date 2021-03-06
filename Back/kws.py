'''
Module : Control Center of Backend 
-----------------------
Features :
1. 选择各种模式
2. 包括训练、测试、启动等

-----------------------
Version :
2018.06.06  第一版

'''

############ 前置module ################
import sys 
import time 
#-------------------------------------#
import kws_train 
import kws_test 
from Tool.kws_error import KWSError
#######################################

if __name__ == '__main__':
    mode = 'test'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    ## 
    if mode == 'train':
        tar = kws_train.Trainer('F:/ASR/THCHS30/data_thchs30/', './log/back', mode='0train')
        tar.corpus.TurnOnOrOffPointer()
        try:
            tar.train(
                0.0001, 0.9, batch_size=8, n_epcho=3000,
                ltype='PH', ltone=True 
                )
        finally:
            tar.log_end()
    elif mode == 'fine':
        tar = kws_train.Trainer('F:/ASR/THCHS30/data_thchs30/', './log/back')
        # tar.corpus.MovePointer(3)
        tar.train(
            0.0001, 0.9, batch_size=3, n_epcho=10,
            ltype='PH', ltone=True, resume=True 
            )
        tar.log_end()

    elif mode == 'test':
        kws_test.TestTargetFileAsWhole(
            'F:/ASR/THCHS30/data_thchs30/0train', './log/Back', 10, 
            ltype='PH', ltone=True 
            )
    elif mode == 'test_frame':
        kws_test.TestTargetFileAsFrame(
            'F:/ASR/THCHS30/data_thchs30/0train', './log/Back', 1, 
            ltype='PH', ltone=True 
            )
    elif mode == 'hi_train':
        tar = kws_train.Trainer('F:/ASR/THCHS30/data_thchs30/', './log/back', mode='A11_train')
        # tar.corpus.TurnOnOrOffPointer()
        try:
            tar.train(
                0.0001, 0.9, batch_size=4, n_epcho=5000,
                ltype='PH', ltone=True#, resume=True  
                )
        finally:
            tar.log_end()
    
    #####
    # end