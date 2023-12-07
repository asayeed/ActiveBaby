#!/usr/bin/env python
# coding: utf-8

# # ActivateBaby - training LM
# based on [How to train a new language model from scratch using Transformers and Tokenizers](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=M1oqh0F6W3ad)
# 

import os
import re
import sys
import time
import json
import random
import pickle
import wandb
import psutil
import nltk
import torch
import transformers

import pandas as pd
import numpy as np

from os.path import join as osj
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from config import args
from lmprobs import TrigramSurprisalSpace
from utils import *

sys.path.insert(1, '/root/xhong/babylm/evaluation_pipeline')

from babylm_eval_Mattes import babylm_evaluation

# Get the current conda environment name from the environment variable
conda_env = os.environ.get('CONDA_DEFAULT_ENV')

# Print the active conda environment
print(f"Active Conda Environment: {conda_env}")

#set the cuda device to avoid distributed training on multiple GPUs
device = 'cuda'

# load arguments with muliple uses

INITIAL_SAMPLE = args['initial_sample']
TSS_SAMPLE_SIZE = args['tss_sample_size'] 
SAMPLE_PER_ITER = args['sample_per_iter'] #try larger sample size (100k, 200k)
c = args['c']
encoder_max_length = args['encoder_max_length']
batch_size = args['batch_size']
surprisal_strategy = args['surprisal_strategy'] # {min, max, random}


# RobertaConfig
vocab_size=args['vocab_size']
max_position_embeddings=args['max_position_embeddings']
num_attention_heads=args['num_attention_heads']
num_hidden_layers=args['num_hidden_layers']
type_vocab_size=args['type_vocab_size']
attention_probs_dropout_prob=args['attention_probs_dropout_prob']
bos_token_id=args['bos_token_id']
classifier_dropout=args['classifier_dropout']
eos_token_id=args['eos_token_id']
hidden_act=args['hidden_act']
hidden_dropout_prob=args['hidden_dropout_prob']
hidden_size=args['hidden_size']
initializer_range=args['initializer_range']
intermediate_size=args['intermediate_size']
layer_norm_eps=args['layer_norm_eps']
position_embedding_type=args['position_embedding_type']


# RobertaTokenizer
tokenizer_path = args['tokenizer_path']
tokenizer_max_len = args['tokenizer_max_len']


# DataCollator
mlm_probability=args['mlm_prob']

# wandb
entity=args['entity']
project=args['project']

# Training Arguments
epoch=args['epoch']
save_steps=args['save_steps']
save_total_limit=args['save_total_limit']
prediction_loss_only=args['prediction_loss_only']
dataloader_num_workers=args['dataloader_num_workers']
learning_rate=args['learning_rate']

# data paths
tss_path=args['tss_path']
train_data_path=args['train_data_path']
dev_data_path=args['val_data_path']
test_data_path=args['test_data_path']

# babyblm eval
baby_lm_evaluation=args['baby_lm_evaluation']

# ## 1. Train a language model from scratch

# 
# **Update:** This section follows along the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) script, using our new [`Trainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) directly. Feel free to pick the approach you like best.
# 
# > We’ll train a RoBERTa-like model, which is a BERT-like with a couple of changes (check the [documentation](https://huggingface.co/transformers/model_doc/roberta.html) for more details).
# 
# As the model is BERT-like, we’ll train it on a task of *Masked language modeling*, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. This is taken care of by the example script.

# ### We'll define the following config for the model
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_position_embeddings,
    num_attention_heads=num_attention_heads,
    num_hidden_layers=num_hidden_layers,
    type_vocab_size=type_vocab_size,
  
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    bos_token_id=bos_token_id,
    classifier_dropout=classifier_dropout,
    eos_token_id=eos_token_id,
    hidden_act=hidden_act,
    hidden_dropout_prob=hidden_dropout_prob,
    hidden_size=hidden_size,
    initializer_range=initializer_range,
    intermediate_size=intermediate_size,
    layer_norm_eps=layer_norm_eps,
    model_type="roberta",
    pad_token_id=1,
    position_embedding_type=position_embedding_type,
    torch_dtype="float32",
    transformers_version="4.29.2",
    use_cache=True,
)

# Now let's re-create our tokenizer in transformers
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=tokenizer_max_len)

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

def process_data_to_model_inputs(batch):
    with torch.no_grad():
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

# Here, as we only have one text file, we don't even need to customize our `Dataset`. We'll just use the `LineByLineDataset` out-of-the-box.

# Like in the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) script, we need to define a data_collator.
# 
# This is just a small helper that will help us batch different samples of the dataset together into an object that PyTorch knows how to perform backprop on.
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability= mlm_probability)


# Name the run and specify output directory
run_name = f"ABRoBERTa_10M_min_200_K_lr_{args['learning_rate']}_{surprisal_strategy}"
output_dir = f"../ckpt/{run_name}"

# Initialize wandb run for logging
if args['wandb_logging']:
    wandb.init(entity=entity, project=project, name=run_name)

# list for storing the most confused examples for each epoch
confused_examples = []

# set training argument

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    num_train_epochs=epoch,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    logging_first_step=True,
    prediction_loss_only=prediction_loss_only,
    dataloader_num_workers=dataloader_num_workers,
    run_name=output_dir.split('/')[-1],
    sharded_ddp=False,
    learning_rate=learning_rate
    )
training_args._n_gpu=1

# We load the data
tss = pickle.load(open(tss_path, "rb"))
train_data_df = pd.read_csv(train_data_path)
dev_data_df = pd.read_csv(dev_data_path)
test_data_df = pd.read_csv(test_data_path)
# (1180291, )
pool = train_data_df['line_idx'].to_numpy()

max_iteration = float(train_data_df.shape[0] - INITIAL_SAMPLE) / SAMPLE_PER_ITER

# We pick out an random inital pool with uniform probability in a temporary directory of n% of the data.
initial_indices = np.random.choice(pool, INITIAL_SAMPLE)
pool = np.delete(pool, initial_indices)
tss.remove_from_space(initial_indices)
sampled_train_data_df = train_data_df.loc[initial_indices,:]

iteration = 0
convergence_criterion_not_met = True

# initialize Trainer without data
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )


while convergence_criterion_not_met: # another miracle
    # dataset = LineByLineTextDataset(
    #     tokenizer=tokenizer,
    #     file_path=training_filename, #REPLACE WITH CURRENT TRAINING SET
    #     block_size=512,
    # )
    dataset = Dataset.from_pandas(sampled_train_data_df)
    dev_dataset = Dataset.from_pandas(dev_data_df.loc[:SAMPLE_PER_ITER,:])
    
    # map data
    train_set = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['Unnamed: 0', 'line_idx', 'token'],
    )
    dev_set = dev_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['Unnamed: 0', 'line_idx', 'token'],
    )

    # ### Finally, we are all set to initialize our Trainer
    trainer.reload_data(train_set, dev_set)

    # ### Start training
    trainer.train()

    # #### 🎉 Save final model (+ tokenizer + config) to disk
    save_path = os.path.join(output_dir, str(iteration))
    trainer.save_model(save_path)
    
    # Move tokenizer files to ckpt directory for evaluation
    copy_tokenizer(tokenizer_path, save_path)
    
    # ### Run the babylm evaluation pipeline
    if baby_lm_evaluation:
        with torch.no_grad():
            babylm_evaluation(model_name=run_name, iteration=iteration)
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        
       
    # Assume a miracle where we know the specific index of the highest perplexity sentence from
    # the training set.    
    # That miracle we will call most_confused_index
    # I.e., for every sentence in the training set, we get the perplexity according to the trained model.
    # find the index of the maximum.
    sampled_indices = np.random.choice(len(train_data_df), TSS_SAMPLE_SIZE)
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
        # TODO: test changing to min?
        
        if surprisal_strategy == 'min':
            max_surprisal_idx = surprisal_array.argmin()
        elif surprisal_strategy == 'max':
            max_surprisal_idx = surprisal_array.argmax()
        elif surprisal_strategy == 'random':
            max_surprisal_idx = np.random.choice(len(surprisal_array), 1)[0]
        else:
            raise ValueError()
          
        most_confused_index = sampled_indices[max_surprisal_idx]
        
        
        # TODO log most confused_index + example
        
        most_confused_example = train_data_df.loc[most_confused_index]
        
        example = {'iteration': f'{iteration}', 'most_confused_index': f'{most_confused_index}' , 'most_confused_example': f'{most_confused_example}'}
        
        confused_examples.append(example)
        
        print('most_confused_index', most_confused_index)

    _, indices, _ = tss.find_index(most_confused_index, k=SAMPLE_PER_ITER) #TODO: k is a hyperparameter
    pool = np.delete(pool, indices)
    # Take things out of the space.
    tss.remove_from_space(indices)
    sampled_train_data_df = train_data_df.loc[indices,:]
    
    iteration += 1
    if iteration > max_iteration or pool.size == 0:
        convergence_criterion_not_met = False
        
        with open(f'../ckpt/{run_name}/confused_examples.json', 'w') as fp:
            json.dump(confused_examples, fp)


    