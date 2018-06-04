'''
Module : Data Input
-----------------------
Features :
1. 能够将音频处理成为特征向量（文件）
2. 训练时，直接获取目标内容
3. 能够根据参数生成：声韵母、音素
4. 能够根据参数生成：mfcc、lfbank

-----------------------
Version :
2018.06.04  第一版

'''

############ 前置module ################
import os 
import numpy as np 
import python_speech_features as psf 
import scipy.io.wavfile as sci_wav
#-------------------------------------#
import sys 
sys.path.append('../')
from ASRcode.Tool.kws_error import KWSError
# from Tool.kws_error import KWSError
#######################################

class DataBase:
    '''
    Basic Tool Set for reading data

    '''
    def __init__(self, kws_log=KWSError('./log/default.log')):
        # log info
        self.logger = kws_log
        self.log_head = self.logger.acquireHeader('kws_data->DataBase')
        return 
    
    ################### Metadata ###########################
    def getFileList(self, dirpath):
        self.audio_files = []
        self.label_files = []
        for (dir_path, _, file_names) in os.walk(dirpath):
            for filename in file_names:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    audio_file_path = os.sep.join([dir_path, filename])
                    label_file_path = os.sep.join([dir_path,'..','data',filename]) 
                    self.audio_files.append(audio_file_path)
                    self.label_files.append(label_file_path + '.trn')
        self.error('Get Files %d' % len(self.audio_files))
        return self.audio_files, self.label_files

    def getLabelSequence(self, ltype='IF'):
        '''
        ltype: 'IF' for 声韵母, 'PH' for 音素
        '''
        # 无论如何都要先获得音节


    ################### Features ############################
    def getMfccArrayFromFile(self, fileName, dimension=40):
        fs, audio = sci_wav.read(fileName)
        return psf.mfcc(audio, fs, numcep=dimension, nfilt=dimension)
    
    def getLogFbankFromFile(self, fileName, dimension=40):
        fs, audio = sci_wav.read(fileName)
        return psf.logfbank(audio, fs, nfilt=dimension)


    def error(self, content):
        self.logger.print(self.log_head, content)
        return 



if __name__ == "__main__":
    #-------------
    test = DataBase()
    test.getFileList('F:/ASR/THCHS30/data_thchs30/dev')



