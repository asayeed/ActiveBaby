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
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ## 1. Train a language model from scratch
# 
# **Update:** This section follows along the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) script, using our new [`Trainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) directly. Feel free to pick the approach you like best.
# 
# > We’ll train a RoBERTa-like model, which is a BERT-like with a couple of changes (check the [documentation](https://huggingface.co/transformers/model_doc/roberta.html) for more details).
# 
# As the model is BERT-like, we’ll train it on a task of *Masked language modeling*, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. This is taken care of by the example script.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"

# ### We'll define the following config for the model
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
    
    attention_probs_dropout_prob=0.1,
    bos_token_id=0,
    classifier_dropout=None,
    eos_token_id=2,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-05,
    model_type="roberta",
    pad_token_id=1,
    position_embedding_type="absolute",
    torch_dtype="float32",
    transformers_version="4.29.2",
    use_cache=True,
)

# Now let's re-create our tokenizer in transformers
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("../tokenizer/ABByteLevelBPE", max_len=512)

# Finally let's initialize our model.
# 
# **Important:**
# 
# As we are training from scratch, we only initialize from a config, not from an existing pretrained model or checkpoint.
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config).to(device)

print(model.num_parameters())
# => 84 million parameters

# ### Now let's build our training Dataset
# 
# We'll build our dataset by applying our tokenizer to our text file.
# 
# Here, as we only have one text file, we don't even need to customize our `Dataset`. We'll just use the `LineByLineDataset` out-of-the-box.

# Like in the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) script, we need to define a data_collator.
# 
# This is just a small helper that will help us batch different samples of the dataset together into an object that PyTorch knows how to perform backprop on.
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# ### Finally, we are all set to initialize our Trainer
from transformers import Trainer, TrainingArguments

from lmprobs import TrigramSurprisalSpace
import pickle
# We pick out an random inital pool with uniform probability in a temporary directory of n% of the data.

tss = pickle.load(open("tss.pkl", "r"))

all_sents = open("../dataset/babylm_10M_sents.txt", "r").readlines()

import random

def sample_pool_random(all_sentences, n):
    selected = random.sample(list(range(len(all_sentences))), n)
    
    corresponding_sents = [all_sentences[x] for x in range(len(all_sentences)) if x in selected]    

    for index in sorted(selected, reverse=True):
        del all_sentences[index] #make sure this behaves as a reference

    return selected, corresponding_sents

def sample_pool_from_selected(all_sentences, selected): 
    corresponding_sents = [all_sentences[x] for x in range(len(all_sentences)) if x in selected] 
    
    for index in sorted(selected, reverse=True):
        del all_sentences[index] #make sure this behaves as a reference

    return corresponding_sents


INITIAL_SAMPLE = 100000
SAMPLE_SIZE = 500

initial_indices, initial_sents = sample_pool_random(pool_sents, all_sents, INITIAL_SAMPLE)
tss.remove_from_space(initial_indices)

TRAININGDIR = "../dataset/trainingsets/"
training_filename = "../dataset/trainingsets/0.txt"
training_file = open(training_filename, "w")
for x in initial_sents:
    trainingfile.write(x)
training_file.close()

iteration = 0
current_sents = initial_sents
while convergence_criterion_not_met: # another miracle        
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=training_filename, #REPLACE WITH CURRENT TRAINING SET
        block_size=512,
    )

    training_args = TrainingArguments(
        output_dir="../ckpt/ABRoBERTa_10M_10ep",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # ### Start training
    trainer.train()

    # #### 🎉 Save final model (+ tokenizer + config) to disk
    trainer.save_model("../ckpt/ABRoBERTa_10M_10ep/final")

    # Assume a miracle where we know the specific index of the highest perplexity sentence from
    # the training pool.
    # That miracle we will call most_confused_index

    _, indices, _ = tss.find_index(most_confused_index, k=500)
    # Take things out of the space.
    tss.remove_from_space(indices)
    additional_sents = sample_pool_from_selected(pool_sents, all_sents, indices)
    
    iteration += 1
    current_sents += additional_sents
    training_filename = f'{TRAININGDIR}/{iteration}.txt'
    training_file = open(training_filename, "w")
    for sent in current_sents:
        training_file.write(sent)
    training_file.close()

    