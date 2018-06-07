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
        tar.train(
            0.0001, 0.9, batch_size=16, n_epcho=8000,
            ltype='PH', ltone=True 
            )
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
        pass 
    
    #####
    # end