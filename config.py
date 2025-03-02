default_args = {
#     'train_data_path': '/root/xhong/babylm/dataset/babylm_10M_sent_tokens.txt', 
    'train_data_path': '/root/xhong/ActiveBaby/elc-bert-10M/data/processed24/cached_128.txt', 
    'val_data_path': '/root/xhong/babylm/dataset',
    'test_data_path': '/root/xhong/babylm/dataset',
    
    'tss_path': 'surprisals_8.pkl',
    'surprisal_space_dim': 8,
    'tokenizer_path': '/root/xhong/babylm/tokenizer/ABByteLevelBPE', 

    #     ## Mostly overwrited by command line input
    # 'model_name': 'no_gt_sos', # hybrid_dis
    # 'log_tag': 'global', # Brief descriptive tag for logdir readibility
    # 'feature_names': ['swin_base'],
}