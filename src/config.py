default_args = {
    ## Mostly overwrited by command line input
    'model_name': 'no_gt_sos', # hybrid_dis
    'log_tag': 'global', # Brief descriptive tag for logdir readibility
    'feature_names': ['swin_base'],
    
    'train_data_path': '/root/xhong/babylm/dataset/babylm_10M_sent_tokens.txt', 
    'val_data_path': '/root/xhong/babylm/dataset',
    'test_data_path': '/root/xhong/babylm/dataset',
    
    'tss_path': 'tss.pkl',
    'tokenizer_path': '/root/xhong/babylm/tokenizer/ABByteLevelBPE', 
}