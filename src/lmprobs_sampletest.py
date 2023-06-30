#!/usr/bin/env python
# coding: utf-8

# # ActivateBaby - training LM
# based on [How to train a new language model from scratch using Transformers and Tokenizers](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=M1oqh0F6W3ad)
# 

import os
import re
import time
from os.path import join as osj
from pathlib import Path
from collections import defaultdict

import nltk
import pandas as pd
from tqdm import tqdm
import numpy as np
#import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from evaluate import load

from lmprobs import TrigramSurprisalSpace
import pickle
# We pick out an random inital pool with uniform probability in a temporary directory of n% of the data.

tss = pickle.load(open("tss.pkl", "rb"))

all_sents = open("../../xhong/babylm/dataset/babylm_10M_sents.txt", "r").readlines()

import random

def sample_pool_random(all_sentences, n):    
    selected = random.sample(list(range(len(all_sentences))), n)

    sent_array = np.array(all_sentences)
    corresponding_sents = list(sent_array[selected])

    new_all = list(np.delete(sent_array, selected))
    
    return selected, corresponding_sents, new_all

def sample_pool_from_selected(all_sentences, selected): 
    sent_array = np.array(all_sentences)
    corresponding_sents = list(sent_array[selected])

    new_all = list(np.delete(sent_array, selected))
    
    return selected, corresponding_sents, new_all

INITIAL_SAMPLE = 100000
SAMPLE_SIZE = 500

initial_indices, initial_sents, all_sents = sample_pool_random(all_sents, INITIAL_SAMPLE)
print(f"Got {initial_indices[0]} which is {initial_sents[0]}")
tss.remove_from_space(initial_indices)

TRAININGDIR = "../dataset/trainingsets/"
training_filename = "../dataset/trainingsets/0.txt"
training_file = open(training_filename, "w")
for x in initial_sents:
    trainingfile.write(x)
training_file.close()