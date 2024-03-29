import os
import sys
from sklearn.neighbors import KDTree
from skimage.transform import resize
from nltk.util import trigrams
from nltk.lm import MLE
import numpy as np
from operator import itemgetter
from nltk.lm.preprocessing import padded_everygram_pipeline
import pickle
from datetime import datetime

from config import default_args

class AbstractSurprisalSpace:
    def __init__(self, dims):
        self.dims = dims

    def fit(self, sequences):
        self.sequences = sequences
        self.train(sequences)

        self.surprisalvecs = [self.surprisalizer_(x) for x in sequences]
        self.currentsurprisalvecs = self.surprisalvecs.copy()
        self.nnfinder = KDTree(self.surprisalvecs)

    def reset_dims(self, newdims):
        self.dims = newdims
        self.surprisalvecs = [self.surprisalizer_(x) for x in self.sequences]
        self.currentsurprisalvecs = self.surprisalvecs.copy()
        # If we reset the dimensionality, we reset the whole space back to the full pool.
        self.nnfinder = KDTree(self.surprisalvecs)
        
    def find_index(self, vec_index, k=5):
        size = self.nnfinder.data.shape[0]
        if k > size:
            return [], [], tuple()
        dists, indices = self.nnfinder.query(self.surprisalvecs[vec_index].reshape(1,-1),
                                      k=k)

        return list(dists[0]), list(indices[0]), itemgetter(*list(indices[0]))(self.surprisalvecs)
    
    # Remove from the stored vectors
    def remove_from_space(self, to_remove):
        #print(f'len {len(self.currentsurprisalvecs)}')
        for index in sorted(to_remove, reverse=True):
            del self.currentsurprisalvecs[index] #make sure this behaves as a reference
    
        self.nnfinder = KDTree(self.currentsurprisalvecs)
        
class TrigramSurprisalSpace(AbstractSurprisalSpace):
    def __init__(self, dims):
        super(TrigramSurprisalSpace, self).__init__(dims)

    def train(self, sequences):
        sequences = [[str(x) for x in list(y)] for y in sequences]
        trainingdata, vocab = padded_everygram_pipeline(3, sequences)
        self.lm = MLE(3)
        self.lm.fit(trainingdata, vocab)

    def surprisalizer_(self, sentence):
        trisent = list(trigrams(sentence))
        surps = np.array([-self.lm.logscore(x[2], [x[0], x[1]]) for x in trisent])
        try:
            resized = resize(surps, (self.dims,))
        except ValueError:
            print("sentence {} trisent {} surps {}".format(sentence, trisent, surps))
        return resized


if __name__ == "__main__":
    tss = TrigramSurprisalSpace(7)
    
    itemfile = open(default_args['train_data_path'], "r")
    tokens = [x[:-1].split(",") for x in itemfile]
    print(f'orig tokens {len(tokens)}')
    #print(tokens[:5000])


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Fit Starting Time =", current_time)
    tss.fit(tokens)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Fit Stopping Time =", current_time)
    
    distances, indices, vectors = tss.find_index(3999)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    
    distances, indices, vectors = tss.find_index(2124)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    
    pickle.dump(tss, open(default_args['tss_path'], "wb"))

    loadtss = pickle.load(open(default_args['tss_path'], "rb"))

    distances, indices, vectors = loadtss.find_index(2124)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    distances, indices, vectors = loadtss.find_index(1111)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    loadtss.reset_dims(10)
    
    distances, indices, vectors = loadtss.find_index(1111)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    
    loadtss.remove_from_space([20,500,550,1024,2048,3333])
    distances, indices, vectors = loadtss.find_index(1024)
    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    