import os
import torch

class Token:
    def __init__(self, str_ = '', count_ = 0):
        self.str = str_
        self.count = count_

class Vocabulary:
    def __init__(self):
        self.UNK = 'UNK' # unkown words
        self.EOS = '<EOS>' # the end-of-sequence token
        self.BOS = '<BOS>' # the beginning-of-sequence token
        self.PAD = '<PAD>' # padding

        self.unkIndex = -1
        self.eosIndex = -1
        self.bosIndex = -1
        self.padIndex = -1
        
        self.tokenIndex = {}
        self.tokenList = []

    def getTokenIndex(self, str):
        if str in self.tokenIndex:
            return self.tokenIndex[str]
        else:
            return self.unkIndex

    def add(self, str, count):
        if str not in self.tokenIndex:
            self.tokenList.append(Token(str, count))
            self.tokenIndex[str] = len(self.tokenList)-1

    def size(self):
        return len(self.tokenList)

    def outputTokenList(self, fileName):
        f = open(fileName, 'w')
        for t in self.tokenList:
            f.write(t.str + '\n')
        f.close()

        
class Data:
    def __init__(self, sourceText_, sourceUnkMap_, targetText_, targetUnkMap_, sourceOrigStr_ = None):
        self.sourceText = sourceText_
        self.sourceUnkMap = sourceUnkMap_

        self.sourceOrigStr = sourceOrigStr_
        
        self.targetText = targetText_
        self.targetUnkMap = targetUnkMap_

        self.smallVoc = None

        
class Corpus:
    def __init__(self, sourceTrainFile = '', sourceOrigTrainFile = '', targetTrainFile = '', sourceDevFile = '', sourceOrigDevFile = '', targetDevFile = '', minFreqSource = 1, minFreqTarget = 1, maxTokenLen = 100000):
        self.sourceVoc = Vocabulary()
        self.targetVoc = Vocabulary()

        self.buildVoc(sourceTrainFile, minFreqSource, source = True)#, maxLen = maxTokenLen)
        self.buildVoc(targetTrainFile, minFreqTarget, source = False)#, maxLen = maxTokenLen)

        self.trainData = self.buildDataset(sourceTrainFile, sourceOrigTrainFile, targetTrainFile, train = True, maxLen = maxTokenLen)
        self.devData = self.buildDataset(sourceDevFile, sourceOrigDevFile, targetDevFile, train = False)

        self.unigramWeight = torch.FloatTensor(self.targetVoc.size()).zero_()
        for d in self.trainData:
            imap = {i:1 for i in d.targetText}
            self.unigramWeight[self.targetVoc.eosIndex] += 1.0
            for i in imap.keys():
                self.unigramWeight[i] += 1.0
        self.unigramWeight /= len(self.trainData)

        self.stat = self.calcStat(sourceTrainFile, targetTrainFile)
        
    def buildVoc(self, fileName, minFreq, source, maxLen = 100000):
        assert os.path.exists(fileName)

        if source:
            voc = self.sourceVoc
        else:
            voc = self.targetVoc
        
        with open(fileName, 'r') as f:
            tokenCount = {}
            unkCount = 0
            eosCount = 0

            for line in f:
                tokens = line.split() # w1 w2 ... \n

                if len(tokens) > maxLen:
                    continue

                eosCount += 1
                
                for t in tokens:
                    if t in tokenCount:
                        tokenCount[t] += 1
                    else:
                        tokenCount[t] = 1

            tokenList = sorted(tokenCount.items(), key = lambda x: -x[1]) # sort by value
            
            for t in tokenList:
                if t[1] >= minFreq:
                    voc.add(t[0], t[1])
                else:
                    unkCount += t[1]

            '''
            Add special tokens
            '''
            voc.add(voc.UNK, unkCount)
            voc.add(voc.BOS, eosCount)
            voc.add(voc.EOS, eosCount)
            voc.add(voc.PAD, 0)

            voc.unkIndex = voc.getTokenIndex(voc.UNK)
            voc.bosIndex = voc.getTokenIndex(voc.BOS)
            voc.eosIndex = voc.getTokenIndex(voc.EOS)
            voc.padIndex = voc.getTokenIndex(voc.PAD)
            
    def buildDataset(self, sourceFileName, sourceOrigFileName, targetFileName, train, maxLen = 100000):
        assert os.path.exists(sourceFileName) and os.path.exists(targetFileName)
        assert os.path.exists(sourceOrigFileName)
        
        with open(sourceFileName, 'r') as fs, open(sourceOrigFileName, 'r') as fsOrig, open(targetFileName, 'r') as ft:
            dataset = []
            
            for (lineSource, lineSourceOrig, lineTarget) in zip(fs, fsOrig, ft):
                tokensSource = lineSource.split() # w1 w2 ... \n
                if train:
                    tokensSourceOrig = None
                else:
                    tokensSourceOrig = lineSourceOrig.split() # w1 w2 ... \n
                tokensTarget = lineTarget.split() # w1 w2 ... \n

                if len(tokensSource) > maxLen or len(tokensTarget) > maxLen or len(tokensSource) == 0 or len(tokensTarget) == 0:
                    continue

                tokenIndicesSource = torch.LongTensor(len(tokensSource))
                unkMapSource = {}
                tokenIndicesTarget = torch.LongTensor(len(tokensTarget))
                unkMapTarget = {}

                for i in range(len(tokensSource)):
                    t = tokensSource[i]
                    tokenIndicesSource[i] = self.sourceVoc.getTokenIndex(t)
                    if tokenIndicesSource[i] == self.sourceVoc.unkIndex:
                        unkMapSource[i] = t

                for i in range(len(tokensTarget)):
                    t = tokensTarget[i]
                    tokenIndicesTarget[i] = self.targetVoc.getTokenIndex(t)
                    if tokenIndicesTarget[i] == self.targetVoc.unkIndex:
                        unkMapTarget[i] = t

                dataset.append(Data(tokenIndicesSource, unkMapSource, tokenIndicesTarget, unkMapTarget, tokensSourceOrig))

                #if len(dataset) == 10000:
                #    break

        return dataset

    def calcStat(self, sourceFileName, targetFileName):
        assert os.path.exists(sourceFileName) and os.path.exists(targetFileName)

        stat = torch.FloatTensor(1000, 1000).fill_(1.0e-08)
        
        with open(sourceFileName, 'r') as fs, open(targetFileName, 'r') as ft:
            dataset = []
            
            for (lineSource, lineTarget) in zip(fs, ft):
                tokensSource = lineSource.split() # w1 w2 ... \n
                tokensTarget = lineTarget.split() # w1 w2 ... \n

                stat[len(tokensSource), len(tokensTarget)] += 1.0

        for i in range(stat.size(0)):
            stat[i] /= stat[i].sum()
                
        return stat

        
    def processBatchInfoNMT(self, batch, device, data_ = None):
        begin = batch[0]
        end = batch[1]
        batchSize = end-begin+1

        '''
        Process source info
        '''
        if data_ is not None:
            data = data_
            batchSize = len(data)
        else:
            data = self.devData[begin:end+1]

        maxLen = len(data[0].sourceText)
        batchInputSource = torch.LongTensor(batchSize, maxLen)
        batchInputSource.fill_(self.sourceVoc.padIndex)
        lengthsSource = []
        
        for i in range(batchSize):
            l = len(data[i].sourceText)
            lengthsSource.append(l)
            
            for j in range(l):
                batchInputSource[i, j] = data[i].sourceText[j]
                
        batchInputSource = batchInputSource.to(device)

        '''
        Process target info
        '''
        data_ = sorted(data, key = lambda x: -len(x.targetText))
            
        maxLen = len(data_[0].targetText)+1 # for BOS or EOS
        batchInputTarget = torch.LongTensor(batchSize, maxLen)
        batchInputTarget.fill_(self.targetVoc.padIndex)
        lengthsTarget = []
        batchTarget = torch.LongTensor(maxLen*batchSize).fill_(-1)
        targetIndexOffset = 0
        tokenCount = 0.0
        
        for i in range(batchSize):
            l = len(data[i].targetText)
            lengthsTarget.append(l+1)
            batchInputTarget[i, 0] = self.targetVoc.bosIndex
            for j in range(l):
                batchInputTarget[i, j+1] = data[i].targetText[j]
                batchTarget[targetIndexOffset+j] = data[i].targetText[j]
            batchTarget[targetIndexOffset+l] = self.targetVoc.eosIndex
            targetIndexOffset += maxLen
            tokenCount += (l+1)

        return batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, data, maxLen

    def processBatchInfoVocGen(self, batchData, train = True, smoothing = True, device = None):
        batchSize = len(batchData)

        genVocInput = []
        genVocInputPos = []

        if batchData[0].targetText is not None and train:
            genVocTarget = torch.FloatTensor(batchSize, self.targetVoc.size()).zero_()

            for i in range(batchSize):
                l = len(batchData[i].targetText)
                genVocTarget[i, self.targetVoc.eosIndex] = 1.0
                for j in range(l):
                    genVocTarget[i, batchData[i].targetText[j]] = 1.0

            if smoothing:
                epsilon = 0.1
                genVocTarget = (1.0-epsilon)*genVocTarget + epsilon*self.unigramWeight

            genVocTarget = genVocTarget.to(device)
        else:
            genVocTarget = None

        offset = 0
        for i in range(batchSize):
            bow = list(set(batchData[i].sourceText))
            l = len(bow)
            genVocInputPos.append(offset)
            offset += l
            genVocInput += bow
            
        genVocInput = torch.LongTensor(genVocInput).to(device)
        genVocInputPos = torch.LongTensor(genVocInputPos).to(device)
        
        return genVocTarget, (genVocInput, genVocInputPos)

