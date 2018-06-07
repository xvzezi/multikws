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
2018.06.05  增加了CMVN
2018.06.06  增加了seq len
            修正了seq len

'''

############ 前置module ################
import os 
import numpy as np 
import python_speech_features as psf 
import scipy.io.wavfile as sci_wav
#-------------------------------------#
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
        # batch info
        self.pointer = 0
        self.pointerOn = True 
        return 

    def LoadMeta(self, dirpath):
        self.loadBasicMeta()                    # 读入基本元数据
        self.getFileList(dirpath)               # 从目标文件夹读取语音文件和符号文件列表

    def LabelSetting(self, ltype, ltone):
        self.getLabelSequence(ltype, ltone)     # 将符号序列读入，并转化为对应类型
        self.transToLabelIndex()                # 将符号序列转化为序号数值，方便训练
    
    def AudioSetting(self, ftype='lfbank', dimension=40, fcmvn=True, fleft=7, fright=2):
        self.ftype = ftype                      # 处理信号的特征法
        self.dimension = dimension              # 信号的维度
        self.fcmvn = fcmvn                      # 是否在窗内进行cmvn处理
        self.fleft = fleft                      # 关键frame左侧的帧数
        self.fright = fright                    # 关键frame右侧的帧数

    def GetDimensionInfo(self):
        feature_dim = self.fleft + self.fright + 1
        feature_dim = feature_dim * self.dimension
        class_num = 0
        if not self.ltone and self.ltype == 'PH':
            # 无调 音素
            class_num = len(self.ch_ph)
        elif self.ltype == 'PH':
            # 有调 音素
            class_num = len(self.ch_ph_tone)
        elif not self.ltone:
            # 无调 声韵母
            class_num = len(self.ch_if)
        else:
            # 有调，声韵母
            class_num = len(self.ch_if_tone)
        return feature_dim, class_num 

    def GetNextBatch(self, batch_size):
        # check 
        if self.pointer + batch_size >= len(self.audio_files):
            self.pointer = 0
        
        # get audio batch
        audio_batch = []
        for i in range(batch_size):
            audio_proc = self.getWindowStackFeaturesFromFile(self.audio_files[self.pointer + i],
                        dimension=self.dimension, ftype=self.ftype, fcmvn=self.fcmvn, fleft=self.fleft, fright=self.fright)
            audio_batch.append(audio_proc)

        # get sequence len 记录真正的长度
        seq_len_batch = []
        for index in audio_batch:
            seq_len_batch.append(len(index))

        # 处理audio batch，使得他们为规则矩阵，否则numpy没法识别
        ## get max batch len 
        max_batch_len = 0
        for audio in audio_batch:
            if len(audio) > max_batch_len:
                max_batch_len = len(audio)
        ## padding
        feature_dim, _ = self.GetDimensionInfo()
        padding_list = [[0 for j in range(feature_dim)]]
        for audio in audio_batch:
            padding_amount = max_batch_len - len(audio)
            for i in range(padding_amount):
                audio += padding_list
        
        # get index batch 
        _, index_batch = self.getIndexSequenceBatch(self.pointer, batch_size)
        if self.pointerOn:
            self.pointer += batch_size
        

        return np.array(audio_batch), index_batch, seq_len_batch

    def GetDecodeDict(self):
        search_dict = self.ch_if_tone          # IF tone
        if not self.ltone and self.ltype == 'PH':
            search_dict = self.ch_ph           # PH no tone
        elif self.ltype == 'PH':
            search_dict = self.ch_ph_tone     # PH tone
        elif not self.ltone:
            search_dict = self.ch_if           # IF no tone 
        
        return search_dict

    def ResetPointer(self):
        self.pointer = 0
        return 
    
    def MovePointer(self, offset):
        self.pointer += offset
        if self.pointer < 0:
            self.pointer = 0
        return 
    
    def TurnOnOrOffPointer(self):
        self.pointerOn = not self.pointerOn
        return 
        
    ################ Metadata & Labels ######################
    # 下面是步骤函数，必须按步骤执行
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
        self.error('IF with tone: class %d' % len(self.ch_if_tone))
        
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
        self.error('IF without tone: class %d' % len(self.ch_if))

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
        self.error('PH without tone: class %d' % len(self.ch_ph))
        
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
        self.error('PH with tone: class %d' % len(self.ch_ph_tone))

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
                self.label_sequences.append(ch_if)
        return self.label_sequences


    def transToLabelIndex(self):
        search_dict = self.ch_if_tone_dict       # IF tone
        if not self.ltone and self.ltype == 'PH':
            search_dict = self.ch_ph_dict       # PH no tone
        elif self.ltype == 'PH':
            search_dict = self.ch_ph_tone_dict  # PH tone
        elif not self.ltone:
            search_dict = self.ch_if_dict       # IF no tone 
        
        self.index_sequences = []
        for label_sequence in self.label_sequences:
            index_sequence = []
            for label in label_sequence:
                index_sequence.append(search_dict[label])
            self.index_sequences.append(index_sequence)
        return self.index_sequences

    # 下面是功能函数
    def getIndexSequenceBatch(self, pointer, batch_size):
        if pointer + batch_size >= len(self.index_sequences):
            return pointer, None 
        loc = pointer
        result = [] 
        for i in range(batch_size):
            result.append(self.index_sequences[loc+i])
        return pointer + batch_size, result 


    ################### Features ############################
    def getMfccArrayFromFile(self, fileName, dimension=40):
        fs, audio = sci_wav.read(fileName)
        return psf.mfcc(audio, fs, numcep=dimension, nfilt=dimension)
    
    def getLogFbankFromFile(self, fileName, dimension=40):
        fs, audio = sci_wav.read(fileName)
        return psf.logfbank(audio, fs, nfilt=dimension)

    def getWindowStackFeaturesFromFile(self, fileName, dimension=40, ftype='lfbank', fcmvn=True, fleft=7, fright=2):
        features = []
        # 基本特征提取
        if ftype == 'lfbank':
            features = self.getLogFbankFromFile(fileName, dimension)
        else:
            features = self.getMfccArrayFromFile(fileName, dimension)
        
        # 取窗口累加
        stacked_features = []
        for i in range(fleft, len(features) - fright):
            cur_window = []
            for j in range(i - fleft, i):
                cur_window.append(features[j])
            for j in range(i, i + fright + 1):
                cur_window.append(features[j])
            
            # cepstral mean and variance normalization
            window = np.array(cur_window)
            mean = np.mean(window, 0)
            if fcmvn:
                variance = np.std(window, 0)
                window = (window - mean) / variance
                cur_window = window
            else:
                cur_window = window - mean 
             
            stacked = []
            cur_window = cur_window.tolist()
            for frame in cur_window:
                stacked += frame 
            stacked_features.append(stacked)
        
        return stacked_features

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
    # test = DataBase()
    # test.loadBasicMeta()
    # print(test.ch_if)
    # print(test.ch_if_tone)
    # print(test.ch_ph)
    # print(test.ch_ph_tone)
    # print('------------------------')
    # test.getFileList('E:/上海交通大学/Lab/ASR/THCHS30/data_thchs30/dev')
    # print(test.audio_files[0])
    # print(test.label_files[0])
    # print('------------------------')
    # test.getLabelSequence(ltype='IF', ltone=True)
    # print(test.label_sequences[0])
    # test.getLabelSequence(ltype='PH', ltone=True)
    # print(test.label_sequences[0])
    # test.getLabelSequence(ltype='IF', ltone=False)
    # print(test.label_sequences[0])
    # test.getLabelSequence(ltype='PH', ltone=False)
    # print(test.label_sequences[0])
    #--------------------
    test = DataBase()
    test.LoadMeta('F:/ASR/THCHS30/data_thchs30/0train')
    print(test.audio_files[0:9])
    test.LabelSetting(ltype='PH', ltone=False)
    test.AudioSetting()
    abatch, lbatch, len_batch = test.GetNextBatch(17)
    print(len(abatch[1]))
    print(len(lbatch[1]))
    print(len_batch)
    abatch, lbatch, len_batch = test.GetNextBatch(4)
    print(len(abatch[1]))
    print(len(lbatch[1]))
    print(len_batch)


