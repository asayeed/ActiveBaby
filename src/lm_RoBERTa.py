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

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../dataset/babylm_10M_sents.txt",
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


