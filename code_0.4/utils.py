from numba import jit
import numpy as np

def buildBatchList(dataSize, batchSize):
    batchList = []
    if dataSize%batchSize == 0:
        numBatch = dataSize//batchSize
    else:
        numBatch = int(dataSize/batchSize)+1
    
    for i in range(numBatch):
        batch = []
        batch.append(i*batchSize)
        if i == numBatch-1:
            batch.append(dataSize-1)
        else:
            batch.append((i+1)*batchSize-1)
        batchList.append(batch)

    return batchList

def flatten(bldTensor):
    return bldTensor.contiguous().view(bldTensor.size(0)*bldTensor.size(1), bldTensor.size(2))

@jit(nopython = True)
def gleu_pre(s1, s2, N):
    totalNgram = 0.0
    matchNgram = 0.0

    s1Len = s1.shape[0]
    s2Len = s2.shape[0]
    
    for n in list(range(N)):
        n += 1
        for i in range(s1Len-n+1):
            totalNgram += 1.0
            for j in range(s2Len-n+1):
                match = True
                for k in range(n):
                    if s1[i+k] != s2[j+k]:
                        match = False
                        break
                if match:
                    matchNgram += 1.0
                    break

    return matchNgram/totalNgram

@jit(nopython = True)
def gleu(s1, s2, N = 1):
    gleu1 = gleu_pre(s1, s2, N)
    gleu2 = gleu_pre(s2, s1, N)

    return (gleu1 if gleu1 < gleu2 else gleu2)


@jit(nopython = True)
def convertBack(batchSize, sampledIndex, output_list):
    for j in range(batchSize):
        index = sampledIndex[j]
        sampledIndex[j] = output_list[j, index]

@jit
def checkTransCondition(batchSize, i, fin, targetWordIndices, targetWordLengths, eosIndex, sampledIndex, eosCount):
    for j in range(batchSize):
        if not fin[j] and targetWordIndices[j, i-1] != eosIndex:
            targetWordLengths[j] += 1
            if sampledIndex[j] == eosIndex:
                eosCount += 1
                fin[j] = True
    return eosCount

@jit
def evalVocGen(batchSize, output_list, batchInputTarget, lengthsTarget, eosIndex):
    recall = 0.0
    for i in range(batchSize):
        imap = set(output_list[i])
        if eosIndex in imap:
            recall += 1.0
        for t in batchInputTarget[i, 1:lengthsTarget[i]]:
            if t in imap:
                recall += 1.0
    return recall

@jit('(int32, int64[:,:], int64[:,:], int32[:], int64[:], int32, int32)')
def convertTargetIndex(batchSize, batchInputTarget, output_list, lengthsTarget, batchTarget, maxTargetLen, eosIndex):
    offset = 0
    for i in range(batchSize):
        imap = set(batchInputTarget[i, 1:1+lengthsTarget[i]-1])
        imap.add(eosIndex)
        
        imap2 = dict()
        l = lengthsTarget[i]
        for j in range(l+1):
            index = output_list[i, j]
            imap2[index] = j
        for j in range(l):
            index = batchTarget[offset+j]
            batchTarget[offset+j] = imap2[index]
        offset += maxTargetLen
