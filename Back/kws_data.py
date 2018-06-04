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
if __name__ == '__main__':
    import sys 
    sys.path.append('../')
    from multikws.Tool.kws_error import KWSError
else:
    from Tool.kws_error import KWSError
#######################################

class DataBase:
    '''
    Basic Tool Set for reading data

    '''
    def __init__(self, kws_log=KWSError('./log/default.log')):
        # log info
        self.logger = kws_log
        self.log_head = self.logger.acquireHeader('kws_data->DataBase')
        # global info 
        self.digit = {
            '1' : 1,
            '2' : 2,
            '3' : 3,
            '4' : 4,
            '5' : 5
        }
        return 

    ################ Metadata & Labels ######################
    def loadBasicMeta(self):
        self.meta_base_path = './Meta/'
        # chinese if with tone 
        self.ch_if_tone = []
        self.ch_if_tone_dict = {}
        with open(self.meta_base_path + 'ch_if_tone.txt') as fp:
            counter = 0
            for line in fp.readlines():
                label = line.rstrip('\n')
                self.ch_if_tone.append(label)
                self.ch_if_tone_dict[label] = counter 
                counter += 1
        
        # chinese if without tone 
        self.ch_if = []
        self.ch_if_dict = {}
        counter = 0
        for label in self.ch_if_tone:
            if self.digit.get(label[-1]) is not None:
                label = label[:-1]
            if self.ch_if_dict.get(label) is None:
                self.ch_if.append(label)
                self.ch_if_dict[label] = counter
                counter += 1

        # chinese ph without tone 
        self.ch_ph = []
        self.ch_ph_dict = {}
        with open(self.meta_base_path + 'ch_ph.txt') as fp:
            counter = 0
            for line in fp.readlines():
                label = line.rstrip('\n')
                self.ch_ph.append(label)
                self.ch_ph_dict[label] = counter 
                counter += 1
        
        # chinese ph's vowel without tone
        self.ch_ph_vowel_dict = {}
        with open(self.meta_base_path + 'ch_ph_vow.txt') as fp:
            for line in fp.readlines():
                label = line.rstrip('\n')
                self.ch_ph_vowel_dict[label] = 1
                
        # chinese ph with tone 
        self.ch_ph_tone = []
        self.ch_ph_tone_dict = {}
        tones = self.digit.keys()
        counter = 0
        for label in self.ch_ph:
            if self.ch_ph_vowel_dict.get(label) is not None:
                for tone in tones:
                    self.ch_ph_tone.append(label+tone)
                    self.ch_ph_tone_dict[label+tone] = counter 
                    counter += 1
            else:
                self.ch_ph_tone.append(label)
                self.ch_ph_tone_dict[label] = counter 
                counter += 1

        # chinese IF to ph dict
        self.ch_if_to_ph_dict = {}
        with open(self.meta_base_path + 'if_to_ph.txt') as fp:
            for line in fp.readlines():
                line = line.rstrip('\n').split('\t')
                tar = line[1:]
                self.ch_if_to_ph_dict[line[0]] = tar 
        

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


    def getLabelSequence(self, ltype='IF', ltone=True):
        '''
        ltype: 'IF' for 声韵母, 'PH' for 音素
        '''
        self.label_sequences = []
        self.ltype = ltype 
        self.ltone = ltone 
        # 无论如何都要先获得音节
        for label_file in self.label_files:
            with open(label_file, encoding='utf-8') as fp:
                chinese = fp.readline()
                syllable = fp.readline()
                ch_if = fp.readline().rstrip('\n').split(' ')
                if not ltone and ltype == 'PH':
                    # 去除音调 转化 音素
                    ch_if = self.convertIfToneToNoTone(ch_if)
                    ch_if = self.convertIfNoToneToPh(ch_if)
                elif ltype == 'PH':
                    # 转化有音调音素
                    ch_if = self.convertIfToneToPh(ch_if)
                elif not ltone:
                    # 去除音调
                    ch_if = self.convertIfToneToNoTone(ch_if)
                    pass  
                self.label_sequences.append(ch_if)
        return self.label_sequences

    ################### Features ############################
    def getMfccArrayFromFile(self, fileName, dimension=40):
        fs, audio = sci_wav.read(fileName)
        return psf.mfcc(audio, fs, numcep=dimension, nfilt=dimension)
    
    def getLogFbankFromFile(self, fileName, dimension=40):
        fs, audio = sci_wav.read(fileName)
        return psf.logfbank(audio, fs, nfilt=dimension)


    ################### Tool ##############################
    # IF : Tone -> No Tone 
    def convertIfToneToNoTone(self, sequence):
        result = []
        for label in sequence:
            if self.digit.get(label[-1]) is not None:
                label = label[:-1]
            result.append(label)
        return result 

    # IF -> PH, No Tone
    def convertIfNoToneToPh(self, sequence):
        result = []
        for label in sequence:
            tars = self.ch_if_to_ph_dict[label]
            if tars[0] == '/':
                continue 
            for tar in tars:
                result.append(tar)
        return result 

    # IF -> PH, Tone 
    def convertIfToneToPh(self, sequence):
        result = []
        for label in sequence:
            tone = ''
            if self.digit.get(label[-1]) is not None:
                tone = label[-1]
                label = label[:-1]
            tars = self.ch_if_to_ph_dict[label]
            if tars[0] == '/':
                continue 
            for tar in tars:
                if self.ch_ph_vowel_dict.get(tar) is not None:
                    tar = tar + tone 
                result.append(tar)
        return result 

    # Logger
    def error(self, content):
        self.logger.print(self.log_head, content)
        return 



if __name__ == "__main__":
    #-------------
    test = DataBase()
    test.loadBasicMeta()
    print(test.ch_if)
    print(test.ch_if_tone)
    print(test.ch_ph)
    print(test.ch_ph_tone)
    print('------------------------')
    test.getFileList('E:/上海交通大学/Lab/ASR/THCHS30/data_thchs30/dev')
    print(test.audio_files[0])
    print(test.label_files[0])
    print('------------------------')
    test.getLabelSequence(ltype='IF', ltone=True)
    print(test.label_sequences[0])
    test.getLabelSequence(ltype='PH', ltone=True)
    print(test.label_sequences[0])
    test.getLabelSequence(ltype='IF', ltone=False)
    print(test.label_sequences[0])
    test.getLabelSequence(ltype='PH', ltone=False)
    print(test.label_sequences[0])




