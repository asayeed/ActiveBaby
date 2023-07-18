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
import random
import pickle

import nltk
import pandas as pd
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from config import default_args
from lmprobs import TrigramSurprisalSpace

# ## 1. Train a language model from scratch

# 
# **Update:** This section follows along the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) script, using our new [`Trainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) directly. Feel free to pick the approach you like best.
# 
# > We’ll train a RoBERTa-like model, which is a BERT-like with a couple of changes (check the [documentation](https://huggingface.co/transformers/model_doc/roberta.html) for more details).
# 
# As the model is BERT-like, we’ll train it on a task of *Masked language modeling*, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. This is taken care of by the example script.

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda"

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

tokenizer = RobertaTokenizerFast.from_pretrained(default_args['tokenizer_path'], max_len=512)

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

# We pick out an random inital pool with uniform probability in a temporary directory of n% of the data.

tss = pickle.load(open(default_args['tss_path'], "rb"))

train_data_df = pd.read_csv("/root/xhong/babylm/dataset/babylm_10M.csv")

INITIAL_SAMPLE = 100000
SAMPLE_SIZE = 50000
MAX_ITERATION = 22
encoder_max_length = 512
batch_size = 64

pool = train_data_df['line_idx'].to_numpy()

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["line"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    return batch

initial_indices = np.random.choice(pool, INITIAL_SAMPLE)
pool = np.delete(pool, initial_indices)
tss.remove_from_space(initial_indices)
sampled_train_data_df = train_data_df.loc[initial_indices,:]

iteration = 0
output_dir = "../ckpt/ABRoBERTa_10M_20iter_BS64/"
convergence_criterion_not_met = True
while convergence_criterion_not_met: # another miracle
    # dataset = LineByLineTextDataset(
    #     tokenizer=tokenizer,
    #     file_path=training_filename, #REPLACE WITH CURRENT TRAINING SET
    #     block_size=512,
    # )
    dataset = Dataset.from_pandas(sampled_train_data_df)
    
    # map train data
    train_set = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['Unnamed: 0', 'line_idx', 'token'],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
    )

    # ### Start training
    trainer.train()

    # #### 🎉 Save final model (+ tokenizer + config) to disk
    save_path = os.path.join(output_dir, str(iteration))
    trainer.save_model(save_path) # TODO: rename checkpoints per iteration
    
    # Assume a miracle where we know the specific index of the highest perplexity sentence from
    # the training set.    
    # That miracle we will call most_confused_index
    # I.e., for every sentence in the training set, we get the perplexity according to the trained model.
    # find the index of the maximum.
    sampled_indices = np.random.choice(len(train_data_df), SAMPLE_SIZE)
    surprisal_by_group = []
    with torch.no_grad():
        for idx in tqdm(sampled_indices):
            line_idx = train_data_df.loc[idx, 'line_idx']
            tokens = train_data_df.loc[idx, 'line']

            # Tokenize the sentences and convert to tensor
            inputs = tokenizer(
                tokens,  
                padding="max_length", 
                truncation=True,
                max_length=encoder_max_length,
                return_tensors='pt').to(device)

            # Perform a forward pass through the model
            outputs = model(**inputs, labels=inputs['input_ids'])

            # The first output is the Cross Entropy loss, calculated per example in the batch
            # Surprisal is the negative log-likelihood, which corresponds to the loss here.
            surprisals = outputs.loss.tolist()
            
            surprisal_by_group.append(surprisals)
        surprisal_array = np.array(surprisal_by_group)
        max_surprisal_idx = surprisal_array.argmax()
        most_confused_index = sampled_indices[max_surprisal_idx]
        
        print('most_confused_index', most_confused_index)

    _, indices, _ = tss.find_index(most_confused_index, k=SAMPLE_SIZE) #TODO: k is a hyperparameter
    pool = np.delete(pool, indices)
    # Take things out of the space.
    tss.remove_from_space(indices)
    sampled_train_data_df = train_data_df.loc[indices,:]
    
    iteration += 1
    if iteration > MAX_ITERATION or pool.size == 0:
        convergence_criterion_not_met = False


    