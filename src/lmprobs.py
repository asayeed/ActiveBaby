import os
import sys
from sklearn.neighbours import KDTree
from skimage.transform import resize
from nltk.util import trigrams
from nltk.lm import MLE
import numpy as np
from operator import itemgetter
from nltk.lm.preprocessing import padded_everygram_pipeline

class AbstractSemanticSpace:
    def __init__(self, dims):
        if isinstance(self, AbstractSurprisalSpace):
            raise TypeError

        self.dims = dims

    def fit(self, sequences):
        self.train(sequences)

        self.suprisalvecs = [self.surprisalizer_(x) for x in sequences]
        self.nnfinder = KDTree(self.surprisalvecs)

    def find(self, vec_index, k=5):
        indices = self.nnfinder.query(self.surprisalvecs[vec_index].reshape(1,-1),
                                      k=k)
        return indices, itemgetter(*vec_index)(self.surprisalvecs)
    
        
class TrigramSurprisalSpace(AbstractSurprisalSpace):
    def __init__(self, dims):
        super(TrigramSurprisalSpace, self).__init__(dims)

    def train(self, sequences):
        sequences = [[str(x) for x in list(y)] for y in sequences]
        trainingdata, vocab = padded_everygram_pipeline(3, sequences)
        self.lm = MLE(3)
        self.lm.fit(trainingdata, vocab)

    def surprisalizer_(self, sentence):
        trisent = trigrams(x)
        return [-self.lm.logscore(x[2], [x[0], x[1]]) for x in trisent]


