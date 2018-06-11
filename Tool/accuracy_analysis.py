'''
Program : Accuracy analysis 
-----------------------
Features :
1. 正确的标签与识别的标签进行对比

-----------------------
Version :
2018.06.09  第一版

'''

############ 前置module ################
import numpy as np 
import json 
from pprint import pprint
#-------------------------------------#
from kws_error import KWSError
#######################################

logger = KWSError('./acc_res.log')
logger_head = logger.acquireHeader('accuracy_analysis')

def log(content):
    logger.print(logger_head, content)
    return 

def edit_distance(word1, word2):
        m=len(word1)+1; n=len(word2)+1
        dp = [[0 for i in range(n)] for j in range(m)]
        delCost = insCost = subCost = 1        # The cost for each operation
         
        for i in range(m):
            dp[i][0]=i
        for j in range(n):
             dp[0][j]=j
         
        for i in range(1,m):
            for j in range(1,n):
                # del                      insert                      same                             sub
                #dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if word1[i-1]==word2[j-1] else 1))
                dp[i][j]=min(dp[i-1][j] + insCost, dp[i][j-1] + delCost, dp[i-1][j-1]+(0 if word1[i-1]==word2[j-1] else subCost))
        # print(dp)
        return dp[m-1][n-1]

def compareOriginAndDecoded(origin, decoded):
    # edit distance
    distance = edit_distance(origin, decoded)
    log('-> Edit distance : {}'.format(distance))

    # phones recognized 
    ## calculate origin data
    origin_amount = 0
    origin_dict = {}
    for phone in origin:
        if origin_dict.get(phone) is None:
            origin_amount += 1
            origin_dict[phone] = 0
    ## calculate decoded data
    decoded_right = 0
    decoded_wrong = 0
    for phone in decoded:
        if origin_dict.get(phone) is None:
            decoded_wrong += 1
        else:
            decoded_right += 1
            origin_dict[phone] += 1
    ## calculate phones not recognized
    recognized_amount = 0
    for key in origin_dict.keys():
        if origin_dict[key] > 0:
            recognized_amount += 1
    ## report 
    acc = (decoded_right - decoded_wrong - abs(len(decoded) - len(origin))) / len(origin)
    log('-> Phones Statistics:')
    log('-> \t Origin: phone kinds: {}, total amount: {}'.format(origin_amount, len(origin)))
    log('-> \t Decode: right kinds: {}, wrong kinds: {}, accuracy: {}'.format(decoded_right, decoded_wrong, acc))
    log('-> \t Recognized phones: {}, recognize rate: {}'.format(recognized_amount, recognized_amount / origin_amount))

    # sequence accuarcy rate 
    log('-> Sequence Accuracy: {}%'.format((100 - distance / len(decoded) * 100)))
    return acc 


def doAccAnalysis(file_path):
    # read the from the json file 
    with open(file_path) as fp:
        result = json.load(fp)
    i = 0
    tmp = []
    while i < len(result):
        pair = (result[i], result[i+1])
        tmp.append(pair)
        i += 2
    result = tmp 

    # begin the analysis
    log('Analysis Begin: {} pairs in total'.format(len(result)))
    counter = 0
    acc = []
    for pair in result:
        origin = pair[0]
        decoded = pair[1]
        log('Pair {}:'.format(counter))
        counter += 1
        # compare the difference 
        acc.append(compareOriginAndDecoded(origin, decoded))
        # break 
    log('Analysis End: Average Acc {}%'.format(sum(acc) / len(acc)))
    log('{}'.format(acc))
    return 

if __name__ == '__main__':
    doAccAnalysis('./acc_res.json') 
    logger.record()