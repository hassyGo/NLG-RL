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
    def __init__(self, in_features, out_features, voc_features, dropoutRate):
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
        super(ValueNet, self).__init__()

        self.outputLayer = nn.Linear(in_features, 1)

    def initWeight(self):
        self.outputLayer.weight.data.uniform_(-0.1, 0.1)
        self.outputLayer.bias.data.zero_()

    def forward(self, input):
        return self.outputLayer(input)

