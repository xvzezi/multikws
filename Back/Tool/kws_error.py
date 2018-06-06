'''
Module : Error ouput
-----------------------
Features :
1. 输出错误信息

-----------------------
Version :
2018.06.04  第一版

'''

class KWSError:
    def __init__(self, path=''):
        self.path = path 
        # 所有的记录
        self.content = []
        self.line_No = 0
        # 行头
        self.headers = []
        self.head_No = 0
        return 

    def acquireHeader(self, head):
        if type(head) is not str:
            print('Warning! Given Header not str. Using Numbers')
            head = str(self.head_No)
        self.headers.append(head)
        self.head_No += 1
        return self.head_No - 1
    
    def print(self, headNo, content):
        tar = "%d %s : %s" % (self.line_No, self.headers[headNo], content) 
        print(tar)
        self.content.append(tar)
        self.line_No += 1
        return 
    
    def record(self):
        with open(self.path, 'w') as fp:
            for line in self.content:
                fp.write(line)
                fp.write('\n')
        self.content = []
        return 

if __name__ == '__main__':
    # test 
    log = KWSError('./Tool/test.log')
    head = log.acquireHeader('Test')
    log.print(head, "Success! Msg Printed!")
    log.print(head, "Test Again")
    head1 = log.acquireHeader('Test Another Header')
    log.print(head1, "Change Header Success!")
    log.print(head, "Changed Back!")
    log.record()
