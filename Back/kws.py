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
        tar = kws_train.Trainer('F:/ASR/THCHS30/data_thchs30/', './log/back')
        tar.train(
            0.001, 0.9, batch_size=16, n_epcho=40,
            ltype='PH', ltone=True 
            )
        tar.log_end()
    elif mode == 'test':
        pass 
    
    #####
    # end