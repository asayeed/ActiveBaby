args = {
    
    # Curriculum learning
    'initial_sample': 100000, # sample size for the first training loop
    'tss_sample_size': 50000, # ?
    'sample_per_iter': 50000, # sample sizes for active curriculum learning
    'c' : 50000, # ?
    'surprisal_strategy': 'max', # {min, max, random}
    
    # babylm evaluation
    'baby_lm_evaluation': True, # {True, False}

    # paths
    'train_data_path': "/root/xhong/babylm/dataset/babylm_10M_fixed.csv", 
    'val_data_path': "/root/xhong/babylm/dataset/babylm_10M_dev.csv",
    'test_data_path': "/root/xhong/babylm/dataset/babylm_10M_test.csv",
    'tss_path': 'tss1.pkl',
    
    # tokenizer
    'tokenizer_path': '/root/xhong/babylm/ActiveBaby/tokenizer/ABByteLevelBPE',
    'tokenizer_max_len': 512,
    
    # wandb
    'wandb_logging': True,
    'entity': 'hiwi_mwxh', # wandb entity
    'project': 'babylm', # wandb project name
    
    # RobertaConfig
    'vocab_size': 52000, # model config vocab size
    'max_position_embeddings': 514,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 1,
    'attention_probs_dropout_prob': 0.1,
    'bos_token_id': 0,
    'classifier_dropout': None,
    'eos_token_id':2,
    'hidden_act': "gelu",
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'layer_norm_eps': 2.5e-06,
    'model_type': "roberta",
    'position_embedding_type': "absolute",
    
    # TrainingArguments
    'epoch': 5, # number of epochs for each training loop
    'encoder_max_length': 512,
    'batch_size': 32,
    'learning_rate': 5e-5,
    'save_steps': 10000,
    'save_total_limit': 2,
    'prediction_loss_only': True,
    'dataloader_num_workers': 1,
    
    #DataCollator
    'mlm_prob': 0.15,
    
    # others
    'log_tag': 'global', # Brief descriptive tag for logdir readibility
}