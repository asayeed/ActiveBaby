import pickle
import os
import sys
import json
from operator import itemgetter
from datetime import datetime

import numpy as np
from sklearn.neighbors import KDTree
from skimage.transform import resize
from nltk.util import trigrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from tqdm import tqdm

from config import default_args

class AbstractSurprisalSpace:
    def __init__(self, dims):
        self.dims = dims

    def fit(self, sequences):
        self.sequences = sequences

        print("Training language model. ")
        vocab = self.train(sequences)
        print("Building surprisal space. ")
        self.surprisalvecs = []
        for x in tqdm(sequences):
            self.surprisalvecs.append(self.surprisalizer_(x))
            
        self.currentsurprisalvecs = self.surprisalvecs.copy()

        print("Building KD Tree. ")
        self.nnfinder = KDTree(self.surprisalvecs)

    def reset_dims(self, newdims):
        self.dims = newdims
        
        self.surprisalvecs = []
        for x in tqdm(self.sequences):
            self.surprisalvecs.append(self.surprisalizer_(x))
            
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
        # print(f'length of surprisal space {len(self.currentsurprisalvecs)}')
        for index in sorted(to_remove, reverse=True):
            # print(index)
            del self.currentsurprisalvecs[index] #make sure this behaves as a reference
    
        self.nnfinder = KDTree(self.currentsurprisalvecs)
        
class TrigramSurprisalSpace(AbstractSurprisalSpace):
    def __init__(self, dims):
        super(TrigramSurprisalSpace, self).__init__(dims)

    def train(self, sequences):
        # sequences = [[str(x) for x in list(y)] for y in sequences]
        # print('sequences', sequences[0])
        trainingdata, vocab = padded_everygram_pipeline(3, sequences)
        # print('trainingdata', trainingdata)
        # for i, g in enumerate(trainingdata):
        #     for v in g:
        #         print(v)
        #     if i > 50: break
        # print('vocab', vocab)
        # for i, v in enumerate(vocab):
        #     print(v)
        #     if i > 50: break
        self.lm = MLE(3)
        self.lm.fit(trainingdata, vocab)
        return vocab

    def surprisalizer_(self, sentence):
        if len(sentence) == 2:
            sentence.append('.')
        if len(sentence) == 1:
            sentence.append('.')
            sentence.append('.')
        if len(sentence) == 0:
            sentence.append('.')
            sentence.append('.')
            sentence.append('.')
        trisent = list(trigrams(sentence))
        surps = np.array([-self.lm.logscore(x[2], [x[0], x[1]]) for x in trisent if len(x)>=3])
        try:
            resized = np.nan_to_num(resize(surps, (self.dims,)))
        except ValueError:
            resized = np.array(['O'])
            print("sentence {} trisent {} surps {}".format(sentence, trisent, surps))
        return resized


if __name__ == "__main__":
    tss = TrigramSurprisalSpace(default_args['surprisal_space_dim'])
    
    itemfile = open(default_args['train_data_path'], "r")
    tokens = [x.split(" ") for x in itemfile]
    print(f'orig tokens {len(tokens)}')
    print(tokens[3999])
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

    # loadtss.reset_dims(7)
    
    # distances, indices, vectors = loadtss.find_index(1111)

    # print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
        
    # pickle.dump(tss, open('tss1.pkl', "wb"))
    
    # loadtss.remove_from_space([20,500,550,1024,2048,3333])
    # distances, indices, vectors = loadtss.find_index(1024)
    # print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    