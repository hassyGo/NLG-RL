import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils

import math
import numpy as np

class Embedding(nn.Module):

    def __init__(self, sourceEmbedDim, targetEmbedDim, sourceVocSize, targetVocSize):
        super(Embedding, self).__init__()

        self.sourceEmbedding = nn.Embedding(sourceVocSize, sourceEmbedDim)
        self.targetEmbedding = nn.Embedding(targetVocSize, targetEmbedDim)

        self.initWeights()

    def initWeights(self):
        initScale = 0.1
        
        self.sourceEmbedding.weight.data.uniform_(-initScale, initScale)
        self.targetEmbedding.weight.data.uniform_(-initScale, initScale)

    def getBatchedSourceEmbedding(self, batchInput):
        return self.sourceEmbedding(batchInput)

    def getBatchedTargetEmbedding(self, batchInput):
        return self.targetEmbedding(batchInput)


class SmallSoftmax(nn.Module):
    def __init__(self, in_features, out_features):
        super(SmallSoftmax, self).__init__()

        self.weight = nn.Embedding(out_features, in_features)
        self.bias = nn.Embedding(out_features, 1)

        self.weight_subset = None
        self.bias_subset = None
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.weight.data.zero_()
        self.bias.weight.data.zero_()

    def setSubset(self, output_list):
        self.weight_subset = self.weight(output_list)
        self.bias_subset = self.bias(output_list)
        
    def forward(self, input, output_list = None, full_softmax = False):
        if output_list is not None:
            self.setSubset(output_list)

        if full_softmax:
            return F.linear(input, self.weight.weight, self.bias.weight.t())
        else:
            return torch.baddbmm(self.bias_subset.transpose(1, 2), input, self.weight_subset.transpose(1, 2))

    def computeLoss(self, output, target):
        return F.cross_entropy(output, target, size_average = False, ignore_index = -1)


class ResBlock(nn.Module):
    def __init__(self, in_features, dropout_rate):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_features, affine = False)
        self.bn2 = nn.BatchNorm1d(in_features, affine = False)
        self.dropout = nn.Dropout(p = dropout_rate)

        self.affine1 = nn.Linear(in_features, in_features)
        self.affine2 = nn.Linear(in_features, in_features)
        self.affineAct = nn.Tanh()
        
        self.reset_parameters()

    def reset_parameters(self):
        initScale = 0.1

        self.affine1.weight.data.uniform_(-initScale, initScale)
        self.affine1.bias.data.zero_()

        self.affine2.weight.data.uniform_(-initScale, initScale)
        self.affine2.bias.data.zero_()

        
    def forward(self, input):
        hidden1 = self.bn1(input)
        hidden1 = self.affineAct(hidden1)
        hidden1 = self.affine1(hidden1)
        
        hidden2 = self.bn2(hidden1)
        hidden2 = self.affineAct(hidden2)
        hidden2 = self.dropout(hidden2)
        hidden2 = self.affine2(hidden2)

        return (hidden2+input)


class VocGenerator(nn.Module):
    def __init__(self, in_features, out_features, voc_features, dropoutRate = 0.4):
        super(VocGenerator, self).__init__()

        self.sourceEmbedding = nn.EmbeddingBag(voc_features, in_features)
        self.resblock = ResBlock(in_features, dropout_rate = dropoutRate)
        self.outputLayer = nn.Linear(in_features, out_features)

        self.reset_parameters()

    def reset_parameters(self):
        initScale = 0.1

        self.sourceEmbedding.weight.data.uniform_(-initScale, initScale)
        
        self.outputLayer.weight.data.zero_()
        self.outputLayer.bias.data.zero_()

    def forward(self, input):
        input = self.sourceEmbedding(input[0], input[1])

        hidden = self.resblock(input)
        output = self.outputLayer(hidden)
        
        return output

    def computeLoss(self, output, target):
        return F.binary_cross_entropy_with_logits(output, target, weight = None, size_average = False)


class BaselineEstimator(nn.Module):
    def __init__(self, in_features):
        super(BaselineEstimator, self).__init__()

        self.outputLayer = nn.Linear(in_features, 1)

    def initWeight(self):
        self.outputLayer.weight.data.uniform_(-0.1, 0.1)
        self.outputLayer.bias.data.zero_()

    def forward(self, input):
        return self.outputLayer(input)


class DecCand:
    def __init__(self, score_ = 0.0, fin_ = False, sentence_ = [], attenIndex_ = []):
        self.score = score_
        self.fin = fin_
        self.sentence = sentence_
        self.attenIndex = attenIndex_

        
class EncDec(nn.Module):

    def __init__(self, sourceEmbedDim, targetEmbedDim, hiddenDim, targetVocSize, useSmallSoftmax, dropoutRate, numLayers):
        super(EncDec, self).__init__()

        self.numLayers = numLayers

        assert self.numLayers == 1 or self.numLayers == 2
        
        self.dropout = nn.Dropout(p = dropoutRate)
        self.embedDropout = nn.Dropout(p = dropoutRate)

        if numLayers == 1:
            dropoutRate = 0.0
            
        self.encoder = nn.LSTM(input_size = sourceEmbedDim, hidden_size = hiddenDim,
                               num_layers = self.numLayers, dropout = dropoutRate, bidirectional = True)

        self.decoder = nn.LSTM(input_size = targetEmbedDim + hiddenDim, hidden_size = hiddenDim,
                               num_layers = self.numLayers, dropout = dropoutRate, bidirectional = False, batch_first = True)

        self.attentionLayer = nn.Linear(2*hiddenDim, hiddenDim, bias = False)
        self.finalHiddenLayer = nn.Linear(3*hiddenDim, targetEmbedDim)
        self.finalHiddenAct = nn.Tanh()
            
        self.softmaxLayer = SmallSoftmax(targetEmbedDim, targetVocSize)

        self.targetEmbedDim = targetEmbedDim
        self.hiddenDim = hiddenDim
        
        self.initWeight()
        
    def initWeight(self):
        initScale = 0.1
        
        self.encoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0.data.zero_()
        self.encoder.bias_hh_l0.data.zero_()
        self.encoder.bias_hh_l0.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1

        self.encoder.weight_ih_l0_reverse.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0_reverse.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0_reverse.data.zero_()
        self.encoder.bias_hh_l0_reverse.data.zero_()
        self.encoder.bias_hh_l0_reverse.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1
        
        self.decoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.decoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.decoder.bias_ih_l0.data.zero_()
        self.decoder.bias_hh_l0.data.zero_()
        self.decoder.bias_hh_l0.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1

        if self.numLayers == 2:
            self.encoder.weight_ih_l1.data.uniform_(-initScale, initScale)
            self.encoder.weight_hh_l1.data.uniform_(-initScale, initScale)
            self.encoder.bias_ih_l1.data.zero_()
            self.encoder.bias_hh_l1.data.zero_()
            self.encoder.bias_hh_l1.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1

            self.encoder.weight_ih_l1_reverse.data.uniform_(-initScale, initScale)
            self.encoder.weight_hh_l1_reverse.data.uniform_(-initScale, initScale)
            self.encoder.bias_ih_l1_reverse.data.zero_()
            self.encoder.bias_hh_l1_reverse.data.zero_()
            self.encoder.bias_hh_l1_reverse.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1

            self.decoder.weight_ih_l1.data.uniform_(-initScale, initScale)
            self.decoder.weight_hh_l1.data.uniform_(-initScale, initScale)
            self.decoder.bias_ih_l1.data.zero_()
            self.decoder.bias_hh_l1.data.zero_()
            self.decoder.bias_hh_l1.data[self.hiddenDim:2*self.hiddenDim].fill_(1.0) # forget bias = 1

        self.attentionLayer.weight.data.zero_()
        
        self.finalHiddenLayer.weight.data.uniform_(-initScale, initScale)
        self.finalHiddenLayer.bias.data.zero_()
        
    def encode(self, inputSource, lengthsSource):
        inputSource = self.embedDropout(inputSource)
        packedInput = nn.utils.rnn.pack_padded_sequence(inputSource, lengthsSource, batch_first = True)
        
        h, (hn, cn) = self.encoder(packedInput) # hn, ch: (layers*direction, B, Ds)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        
        if self.numLayers == 1:
            hn = (hn[0]+hn[1]).unsqueeze(0)
            cn = (cn[0]+cn[1]).unsqueeze(0)
        else:
            hn0 = (hn[0]+hn[1]).unsqueeze(0)
            cn0 = (cn[0]+cn[1]).unsqueeze(0)
            hn1 = (hn[2]+hn[3]).unsqueeze(0)
            cn1 = (cn[2]+cn[3]).unsqueeze(0)

            hn = torch.cat((hn0, hn1), dim = 0)
            cn = torch.cat((cn0, cn1), dim = 0)

        return h, (hn, cn)

    def forward(self, inputTarget, lengthsTarget, lengthsSource, hidden0Target, sourceH, output_list = None):
        batchSize = sourceH.size(0)
        maxLen = lengthsTarget[0]
        
        for i in range(batchSize):
            maxLen = max(maxLen, lengthsTarget[i])
        
        finalHidden = Variable(inputTarget.data.new(batchSize, maxLen, self.targetEmbedDim), requires_grad = False)

        prevFinalHidden = Variable(inputTarget.data.new(batchSize, 1, self.targetEmbedDim).zero_(), requires_grad = False)

        newShape = sourceH.size(0), sourceH.size(1), self.hiddenDim # (B, Ls, Dt)
        sourceHtrans = utils.flatten(sourceH) # (B*Ls, Ds)
        sourceHtrans = self.attentionLayer(sourceHtrans) # (B*Ls, Dt)
        sourceHtrans = sourceHtrans.view(*newShape).transpose(1, 2) # (B, Dt, Ls)
        
        for i in range(maxLen):
            ei = self.embedDropout(inputTarget[:, i, :].unsqueeze(1))
            xi = torch.cat((ei, prevFinalHidden), dim = 2)
            hi, hidden0Target = self.decoder(xi, hidden0Target) # hi: (B, 1, Dt)

            if self.numLayers != 1:
                hi = hidden0Target[0][0]+hidden0Target[0][1]
                hi = hi.unsqueeze(1)

            attentionScores_ = torch.bmm(hi, sourceHtrans).transpose(1, 2) # (B, Ls, 1)

            attentionScores = attentionScores_.data.new(attentionScores_.size()).fill_(-1024.0)
            for j in range(batchSize):
                attentionScores[j, :lengthsSource[j]].zero_()
            attentionScores = Variable(attentionScores, requires_grad = False)
            attentionScores += attentionScores_

            attentionScores = attentionScores.transpose(1, 2) # (B, 1, Ls)
            attentionScores = F.softmax(attentionScores.transpose(0, 2)).transpose(0, 2)

            contextVec = torch.bmm(attentionScores, sourceH) # (B, 1, Ds)

            prevFinalHidden = torch.cat((hi, contextVec), dim = 2) # (B, 1, Ds+Dt)
            prevFinalHidden = self.dropout(prevFinalHidden)
            prevFinalHidden = self.finalHiddenLayer(prevFinalHidden)
            prevFinalHidden = self.finalHiddenAct(prevFinalHidden)
            prevFinalHidden = self.dropout(prevFinalHidden)

            finalHidden[:, i, :] = prevFinalHidden

        if output_list is None:
            finalHidden = utils.flatten(finalHidden)
            output = self.softmaxLayer(finalHidden, full_softmax = True)
        else:
            output = self.softmaxLayer(finalHidden, output_list)
            output = utils.flatten(output)

        return output

    
    def sample(self, bosIndex, eosIndex, lengthsSource, targetEmbedding, sourceH, hidden0Target, useSmallSoftmax = False, fullSoftmax = False, output_list = None, train = False, greedyProb = 1.0, maxGenLen = 100):
        batchSize = sourceH.size(0)
        i = 1
        eosCount = 0
        targetWordIndices = Variable(torch.LongTensor(batchSize, maxGenLen).fill_(bosIndex), requires_grad = False, volatile = not train).cuda()
        sampledIndices = Variable(torch.LongTensor(batchSize, maxGenLen).fill_(bosIndex), requires_grad = False, volatile = not train).cuda()
        attentionIndices = targetWordIndices.data.new(targetWordIndices.size())
        targetWordLengths = torch.LongTensor(batchSize).fill_(0)
        fin = [False]*batchSize

        newShape = sourceH.size(0), sourceH.size(1), hidden0Target[0].size(2) # (B, Ls, Dt)
        sourceHtrans = utils.flatten(sourceH) # (B*Ls, Ds)
        sourceHtrans = self.attentionLayer(sourceHtrans) # (B*Ls, Dt)
        sourceHtrans = sourceHtrans.view(*newShape).transpose(1, 2) # (B, Dt, Ls)

        if train:
            finalHiddenAll = Variable(sourceH.data.new(batchSize, maxGenLen, self.targetEmbedDim), requires_grad = False) # (B, Lt, Dt)

        prevFinalHidden = Variable(sourceH.data.new(batchSize, 1, self.targetEmbedDim).zero_(), requires_grad = False, volatile = not train) 
        
        while (i < maxGenLen) and (eosCount < batchSize):
            inputTarget = targetEmbedding(targetWordIndices[:, i-1].unsqueeze(1))
            inputTarget = self.embedDropout(inputTarget)
            xi = torch.cat((inputTarget, prevFinalHidden), dim = 2)
            hi, hidden0Target = self.decoder(xi, hidden0Target) # hi: (B, 1, Dt)

            if self.numLayers != 1:
                hi = hidden0Target[0][0]+hidden0Target[0][1]
                hi = hi.unsqueeze(1)

            attentionScores_ = torch.bmm(hi, sourceHtrans).transpose(1, 2) # (B, Ls, 1)

            attentionScores = attentionScores_.data.new(attentionScores_.size()).fill_(-1024.0)
            for j in range(batchSize):
                attentionScores[j, :lengthsSource[j]].zero_()
            attentionScores = Variable(attentionScores, requires_grad = False, volatile = not train)
            attentionScores += attentionScores_

            attentionScores = attentionScores.transpose(1, 2) # (B, 1, Ls)
            attentionScores = F.softmax(attentionScores.transpose(0, 2)).transpose(0, 2)

            if not train:
                attnProb, attnIndex = torch.max(attentionScores, dim = 2)
                for j in range(batchSize):
                    attentionIndices[j, i-1] = attnIndex.data[j, 0]

            contextVec = torch.bmm(attentionScores, sourceH) # (B, 1, Ds)
            finalHidden = torch.cat((hi, contextVec), 2) # (B, 1, Dt+Ds)
            finalHidden = self.dropout(finalHidden)
            finalHidden = self.finalHiddenLayer(finalHidden)
            finalHidden = self.finalHiddenAct(finalHidden)
            finalHidden = self.dropout(finalHidden)
            prevFinalHidden = finalHidden # (B, 1, Dt)

            if train:
                finalHiddenAll[:, i-1, :] = prevFinalHidden

            if not useSmallSoftmax:
                finalHidden = utils.flatten(finalHidden)
                output = self.softmaxLayer(finalHidden, full_softmax = True)
            else:
                output = self.softmaxLayer(finalHidden, full_softmax = fullSoftmax)
                output = utils.flatten(output)

            rnd = torch.FloatTensor(1).uniform_(0.0, 1.0)[0]
            
            if rnd <= greedyProb:
                maxProb, sampledIndex = torch.max(output, dim = 1)
                sampledIndices[:, i].data.copy_(sampledIndex.data)
                
                if useSmallSoftmax and not fullSoftmax:
                    sampledIndex = sampledIndex.cpu()
                    utils.convertBack(batchSize, sampledIndex.data.numpy(), output_list.data.numpy())
                    
                targetWordIndices.data[:, i].copy_(sampledIndex.data)
                sampledIndex = sampledIndex.data
            else:
                output = F.softmax(output)
                sampledIndex = torch.multinomial(output.data, num_samples = 1).squeeze(1)
                sampledIndices[:, i].data.copy_(sampledIndex)
                
                if useSmallSoftmax and not fullSoftmax:
                    sampledIndex = sampledIndex.cpu()
                    utils.convertBack(batchSize, sampledIndex.numpy(), output_list.data.numpy())
                    
                targetWordIndices.data[:, i].copy_(sampledIndex)

            eosCount = utils.checkTransCondition(batchSize, i, fin, targetWordIndices.cpu().data.numpy(), targetWordLengths, eosIndex, sampledIndex.cpu().numpy(), eosCount)
            
            i += 1

        targetWordIndices = targetWordIndices[:, 1:i] # i-1: no EOS
        sampledIndices = sampledIndices[:, 1:i] # i-1: no EOS
        
        if train:
            finalHiddenAll = finalHiddenAll[:, 0:i-1, :]
            output = self.softmaxLayer(finalHiddenAll, full_softmax = not useSmallSoftmax) # (B, Lt, K)
            output = F.log_softmax(output.transpose(0, 2)).transpose(0, 2) # the same size
            
            outputSize = output.size()
            output = utils.flatten(output)
            sampledIndices = sampledIndices.contiguous().view(sampledIndices.size(0)*sampledIndices.size(1))

            logProbs = output.gather(dim = 1, index = sampledIndices.view(-1, 1))
            logProbs = logProbs.view(outputSize[0], outputSize[1])
            
            finalHiddenAll = Variable(finalHiddenAll.data.new(finalHiddenAll.size()).copy_(finalHiddenAll.data), requires_grad = False)

            return targetWordIndices, list(targetWordLengths), logProbs, finalHiddenAll
        else:
            return targetWordIndices, list(targetWordLengths), attentionIndices

