# coding=utf-8
import pickle
import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import wandb

from tokenizers import Tokenizer
from pre_training.lamb import Lamb
from pre_training.config import BertConfig

from models.model_elc_bert_base import Bert

from models.elcbert_hf import LtgBertPreTrainedModel

from pre_training.utils import (
    cosine_schedule_with_warmup,
    is_main_process,
    get_rank,
    seed_everything,
    get_world_size,
)
from pre_training.dataset import Dataset

# Dependency for ACLM
from config import default_args
from lmprobs import TrigramSurprisalSpace


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path",
        default="./data/processed24/cached_128.txt",
        type=str,
        help="The input data dir. Should contain .hdf5 files for the task.",
    )
    parser.add_argument(
        "--config_file",
        default="./configs/small.json",
        type=str,
        help="The BERT model config",
    )
    parser.add_argument(
        "--output_name",
        default="./checkpoints/ACLM_elc_bert_base_small_24",
        type=str,
        help="The output directory where the model checkpoints \
            will be written.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./tokenizers/tokenizer_small_10M_24.json",
        type=str,
        help="The vocabulary the BERT model will train on.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to a previous checkpointed training state.",
    )

    # Other parameters
    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="The optimizer to use during pre-training (lamb or adamw).",
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        type=str,
        help="(Not implemented)The learning scheduler to use during training (cosine).",
    )
    parser.add_argument(
        "--seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece \
            tokenization. Sequences longer than this will be truncated, \
            and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Total batch size for training per GPUs and per \
            grad accumulation step.",
    )
    parser.add_argument(
            "--learning_rate",
        default=0.005,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--max_steps",
        default=31250 // 4,
        type=int,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--long_after",
        default=0.9,
        type=float,
        help="The fraction of steps after which to quadruple the sequence length.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.016,
        type=float,
        help="Proportion of training to perform linear learning rate warmup \
            for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--log_freq", type=int, default=10, help="frequency of logging loss."
    )
    parser.add_argument(
        "--mask_p", default=0.15, type=float, help="Masking probability."
    )
    parser.add_argument(
        "--short_p", default=0.1, type=float, help="Short sequence probability."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.4,
        type=float,
        help="Fraction of weight decay to apply. (Should be between 0 and 1)",
    )
    parser.add_argument(
        "--max_gradient",
        default=2.0,
        type=float,
        help="Max value for gradient clipping.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        default=4,
        type=int,
        help="The number of gradient accumulation steps to do.",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0,
        type=float,
        help="The label smoothing to apply to apply to cross-entropy.",
    )
    
    parser.add_argument(
        "--wandb_entity", type=str, default="tony-xudong-hong", help="Your WANDB username/entity."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="activebaby", help="WANDB project name."
    )
    parser.add_argument(
        "--wandb_name", type=str, default="ACLM-ELC-BERT", help="WANDB run name."
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None, help="WANDB run id."
    )


    # ACLM arguments
    parser.add_argument(
        "--aclm_csv_path",
        default="./data/processed24/cached_128.csv",
        type=str,
        help="The csv dir. for ACLM sampler.",
    )
    parser.add_argument(
        "--aclm_tss_path",
        default="./surprisals_128.pkl",
        type=str,
        help="The tss dir. for ACLM sampler.",
    )
    parser.add_argument(
        "--aclm_init_size",
        default=64,
        type=int,
        help="The number of intial sample batches.",
    )
    parser.add_argument(
        "--aclm_tss_sample_size",
        default=64,
        type=int,
        help="The number of sample batches to compute surprisals.",
    )
    parser.add_argument(
        "--aclm_sample_per_iter",
        default=32,
        type=int,
        help="The number of sample batches per iteration. ",
    )
    
    args = parser.parse_args()

    return args

@torch.no_grad()
def log_parameter_histograms(model, step):
    for name, param in model.named_parameters():
        wandb.log(
            {
                f"parameters/norm_{name}": torch.linalg.norm(param.data).cpu().item(),
                f"parameters/std_{name}": param.data.std().cpu().item(),
            },
            step=step,
            commit=False,
        )
        if param.requires_grad and param.grad is not None:
            wandb.log(
                {
                    f"gradients/norm_{name}": torch.linalg.norm(param.grad)
                    .cpu()
                    .item(),
                    f"gradients/std_{name}": param.grad.std().cpu().item(),
                },
                step=step,
                commit=False,
            )
        if "prev_layer_weights" in name:
            d = F.softmax(param.data.cpu(), dim=-1).numpy()
            param_dict = {f"layer_weights/{name}_{i}": d[i] for i in range(len(d))}
            wandb.log(param_dict, step=step, commit=False)


def setup_training_not(args):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    # world_size = int(os.environ["WORLD_SIZE"])
    world_size = get_world_size()
    # rank = int(os.environ["SLURM_PROCID"])
    # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    gpus_per_node = torch.cuda.device_count()
    # assert gpus_per_node == torch.cuda.device_count()
    print(
       f"Hello from rank {rank} of {world_size} on {gethostname()} where \
           there are {gpus_per_node} allocated GPUs per node.",
       flush=True,
    )

    rank = torch.cuda.device_count()
    seed_everything(args.seed + rank)

    # torch.distributed.init_process_group(backend='nccl',
    #                                      init_method='env://',
    #                                      rank = torch.cuda.device_count(),
    #                                      world_size = 1)

    torch.distributed.init_process_group(backend = "nccl", rank = rank, world_size = world_size)

    if rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    print(f"RCCL started on device {device}", flush=True)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")

    if is_main_process():
        tok_per_batch = args.batch_size * args.seq_length
        print(
            f"Training for {args.max_steps:,} steps with {get_world_size()} \
                GPUs"
        )
        print(
            f"In total, the model will be trained on 'steps'\
        ({args.max_steps:,}) x 'GPUs'({get_world_size()}) x \
        'batch_size'({args.batch_size:,}) x 'seq_len'\
        ({args.seq_length:,}) = \
        {args.max_steps * get_world_size() * tok_per_batch:,} \
        subword instances"
        )

    args.device_max_steps = args.max_steps

    if args.wandb_id:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            # id=args.wandb_id,
            config=args,
            resume="auto",
            allow_val_change=True,
            reinit=True,
        )

    return device, local_rank

def setup_training(args):
    rank = torch.cuda.device_count()
    seed_everything(args.seed + rank)
    gpus_per_node = torch.cuda.device_count()

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    device = torch.device("cuda", local_rank)
    
    if is_main_process():
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            id=args.wandb_id,
            config=args,
            resume="auto",
            allow_val_change=True,
            reinit=True,
        )
        
    return device, local_rank

def prepare_model_and_optimizer(args, device, checkpoint):
    config = BertConfig(args.config_file)
    model = Bert(config, args.activation_checkpointing)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    no_decay = ["bias", "layer_norm", "embedding", "prev_layer_weights"]
    high_no = ["res"]
    decay_params = [
        (n, p)
        for n, p in model.named_parameters()
        if (not any(nd in n for nd in no_decay) and not any(hn in n for hn in high_no))
    ]
    no_decay_params = [
        (n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
    ]
    high_no_decay_params = [
        (n, p) for n, p in model.named_parameters() if any(hn in n for hn in high_no)
    ]
    optimizer_grouped_parameters = [
        {"params": [p for _, p in decay_params], "weight_decay": args.weight_decay},
        {"params": [p for _, p in no_decay_params], "weight_decay": 0.0},
        {
            "params": [p for _, p in high_no_decay_params],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 1,
        },
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print()
        print("Parameters with no weight decay and high learning rate:")
        for n, _ in high_no_decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    scheduler = cosine_schedule_with_warmup(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        args.max_steps,
        0.1,
    )

    # model = DistributedDataParallel(
    #    model,
    #    device_ids=[0],
    #    bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
    #    broadcast_buffers=False,
    #    gradient_as_bucket_view=True,
    #    static_graph=True,
    # )

    grad_scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    return model, config, optimizer, scheduler, grad_scaler

def training_epoch(
    model,
    tokenizer,
    data,
    optimizer,
    scheduler,
    grad_scaler,
    global_step,
    epoch,
    args,
    device,
    max_local_steps,
):
    seed = args.seed + get_rank() + epoch * get_world_size()
    train_dataloader = create_train_dataloader(data, args, global_step, seed)

    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0
    avg_accuracy = 0

    if is_main_process():
        # print(" ===> is_main_process yes")
        current_step = global_step * args.gradient_accumulation
        max_steps = args.max_steps * args.gradient_accumulation
        train_iter = tqdm(
            train_dataloader,
            desc="Train iteration",
            initial=current_step,
            total=max_steps,
        )
    else:
        # print(" ===> is_main_process no")
        train_iter = train_dataloader


    for local_step, batch in enumerate(train_iter):
        # print("===> local_step", local_step)
        input_ids, attention_mask, target_ids = [
            t.to(device, non_blocking=True) for t in batch
        ]
        # print("===> input_ids", input_ids, attention_mask, target_ids)

        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.amp.autocast(device_type='cuda'):
            model.train()
            # print(" ")
            # print("debug print ==>", input_ids.size(), attention_mask.size(), target_ids.size(), )
            
            prediction, masked_lm_labels = model(input_ids, attention_mask, target_ids)
            prediction = torch.index_select(
                prediction.flatten(0, 1),
                0,
                torch.nonzero(masked_lm_labels.flatten() != -100).squeeze(),
            )
            
            target_ids = target_ids.flatten()
            target_ids = target_ids[target_ids != -100]
            
            loss = F.cross_entropy(
                prediction, target_ids, label_smoothing=args.label_smoothing
            )
            loss /= args.gradient_accumulation
            total_loss += loss.item()
            
        with torch.no_grad():
            accuracy = (prediction.argmax(-1) == target_ids).float().mean()
            avg_accuracy += accuracy.item() / args.gradient_accumulation

        grad_scaler.scale(loss).backward()
        if (local_step + 1) % args.gradient_accumulation == 0:
            grad_scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

            return_value = grad_scaler.step(optimizer)
            grad_scaler.update()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if return_value is None:
                continue

            scheduler.step()

            if is_main_process():
                train_iter.set_postfix_str(
                    f"loss: {total_loss:.2f}, \
                                    accuracy: {avg_accuracy * 100.0:.2f}, \
                                    grad_norm: {grad_norm:.2f}, \
                                    lr: {optimizer.param_groups[0]['lr']:.5f}"
                )

                total_loss = 0
                avg_accuracy = 0

        if (
            global_step == int(args.max_steps * args.long_after)
            and (local_step + 1) % args.gradient_accumulation == 0
        ):
            optimizer.zero_grad(set_to_none=True)
            return global_step

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps or local_step >= max_local_steps - 1:
            optimizer.zero_grad(set_to_none=True)
            return global_step

    optimizer.zero_grad(set_to_none=True)
    return global_step

# TODO: change to HF compatible checkpoints
def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    bs = 'B'+ str(args.batch_size)
    spi = 'S'+ str(args.aclm_sample_per_iter)
    dimension = 'D'+args.aclm_tss_path.split('.')[-2].split('_')[-1]
    output_dir = '_'.join([args.output_name, bs, spi, dimension])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    checkpoint_path = os.path.join(output_dir, f"model_epoch{epoch}.bin")
    if is_main_process():
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path,
        )

    return checkpoint_path

def load_dataset(args, tokenizer, device, dataset=None):
    seq_length = (
        args.seq_length * 4
        if global_step >= int(args.max_steps * args.long_after)
        else args.seq_length
    )
    # Load dataset for df
    if type(dataset) == pd.DataFrame: 
        train_data = Dataset(
            dataset,
            get_rank(),
            get_world_size(),
            tokenizer,
            seq_length,
            args.mask_p,
            args.short_p,
        )
    # Load dataset for file
    else:
        train_data = Dataset(
            args.input_path.format(sequence_length=seq_length),
            get_rank(),
            get_world_size(),
            tokenizer,
            seq_length,
            args.mask_p,
            args.short_p,
        )
        
    # print(f" ==> Loaded training file", flush=True)

    batch_size = (
        args.batch_size // 4
        if global_step > args.max_steps * args.long_after
        else args.batch_size
    )
    min_length = torch.tensor(
        len(train_data) // batch_size, dtype=torch.long, device=device
    )
    #torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    return train_data, min_length



def create_train_dataloader(data, args, global_step, seed):
    batch_size = (
        args.batch_size // 4
        if global_step >= int(args.max_steps * args.long_after)
        else args.batch_size
    )
    train_dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=7 - 1,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader

if __name__ == "__main__":
    print("# **************** start main **********")
    args = parse_arguments()
    args.mixed_precision = True
    args.activation_checkpointing = False

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint["args"]
        initial_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0

    tokenizer = Tokenizer.from_file(args.vocab_path)
    device, local_rank = setup_training(args)

    print("device, local_rank: ", device, local_rank)
    
    model, config, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(
        args, device, checkpoint)
    
    print("# **************** prepare ACLM **********")
    
    # ACLM hyper-parameters
    INITIAL_SAMPLE = args.aclm_init_size * args.batch_size
    TSS_SAMPLE_SIZE = args.aclm_tss_sample_size * args.batch_size
    SAMPLE_PER_ITER = args.aclm_sample_per_iter * args.batch_size
    
    # Load ACLM data
    train_data_df = pd.read_csv(args.aclm_csv_path)
    max_iteration = float(train_data_df.shape[0] - INITIAL_SAMPLE) / SAMPLE_PER_ITER

    print("TSS_SAMPLE_SIZE: ", TSS_SAMPLE_SIZE)
    print("SAMPLE_PER_ITER: ", SAMPLE_PER_ITER)
    print("max_iteration: ", max_iteration)

    for epoch in count(initial_epoch):
        print("## ************ start ACLM epoch: {}**********".format(epoch), )
        pool = train_data_df['line_idx'].to_numpy()
        tss = pickle.load(open(args.aclm_tss_path, "rb"))
        initial_indices = np.random.choice(pool, INITIAL_SAMPLE, replace=False)
        
        pool = np.delete(pool, initial_indices)
        tss.remove_from_space(initial_indices)
        sampled_train_data_df = train_data_df.loc[initial_indices,:]
        
        split = 0
        convergence_criterion_not_met = True
        while convergence_criterion_not_met:
            print("### ******** start ACLM split: {} ********".format(split))
    
            print("****** start load_dataset **********")
            train_data, min_length = load_dataset(args, tokenizer, device, sampled_train_data_df)
            # train_data, min_length = load_dataset(args, tokenizer, device)
            print("****** data is loaded **********")
            print('min_length', min_length)
            print('global_step', global_step)
            if min_length.item() == 0:
                break
    
            # if global_step == int(args.max_steps * args.long_after):
            #     train_data, min_length = load_dataset(args, tokenizer, device)
    
            global_step = training_epoch(
                model,
                tokenizer,
                train_data,
                optimizer,
                scheduler,
                grad_scaler,
                global_step,
                epoch,
                args,
                device,
                min_length,
            )
    
            print("#### **** start ACLM surprisal computing ******")
            ### ACML steps
            # Assume a miracle where we know the specific index of the highest perplexity sentence from
            # the training set.    
            # That miracle we will call most_confused_index
            # I.e., for every sentence in the training set, we get the perplexity according to the trained model.
            # find the index of the maximum.
            if TSS_SAMPLE_SIZE > 0:
                sampled_indices = np.random.choice(len(train_data_df), TSS_SAMPLE_SIZE, replace=False)
            else:
                sampled_indices = np.random.choice(len(train_data_df), len(train_data_df), replace=False)
            surprisal_by_group = []
            sampled_df = train_data_df.loc[sampled_indices,:]
            sampled_data, sampled_min_length = load_dataset(args, tokenizer, device, sampled_df)
            sampled_seed = args.seed + get_rank() + epoch * get_world_size()
            sampled_iter = DataLoader(
                sampled_data,
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=7 - 1,
                generator=torch.Generator().manual_seed(sampled_seed),
                drop_last=True,
                pin_memory=True,
            )
            model.eval()
    
            with torch.no_grad():
                for local_step, batch in enumerate(sampled_iter):
                    # print("===> local_step", local_step)
                    # get ids
                    input_ids, attention_mask, target_ids = [
                        t.to(device, non_blocking=True) for t in batch
                    ]
                    # print("===> input_ids", input_ids, attention_mask, target_ids)
                    input_ids, target_ids = input_ids.t(), target_ids.t()
                    
                    with torch.amp.autocast(device_type='cuda'):
                        # model.eval()
                        # print("debug print ==>", input_ids.size(), attention_mask.size(), target_ids.size(), )
                        # Perform a forward pass through the model
    
                        prediction, masked_lm_labels = model(input_ids, attention_mask, target_ids)
                        
                        seq_length = prediction.size(1) // args.batch_size  # Calculate sequence length
                        
                        # print("debug print ==> before ", prediction.size(), target_ids.size(), )
                        prediction = prediction.flatten(0, 1)
                        target_ids = target_ids.flatten()
                        # print("debug print ==> after ", prediction.size(), target_ids.size(), )
    
                        loss = F.cross_entropy(
                            prediction, target_ids, reduction='none', label_smoothing=args.label_smoothing
                        )
                        loss = loss.view(args.batch_size, -1)
                        
                        # Take the mean along the sequence length for each entry in the batch
                        mean_loss = loss.mean(dim=1)
                        
                        # print('loss.size()', loss.size())
                        # print('mean_loss.size()', loss.size())
                        
                        # The first output is the Cross Entropy loss, calculated per example in the batch
                        # Surprisal is the negative log-likelihood, which corresponds to the loss here.
                        surprisals = mean_loss.tolist()
                    surprisal_by_group += surprisals
                    
            print("#### **** end ACLM surprisal computing ******")
    
            surprisal_array = np.array(surprisal_by_group)
            print('surprisal_array.shape', surprisal_array.shape)
            
            max_surprisal_idx = surprisal_array.argmax()
            most_confused_index = sampled_indices[max_surprisal_idx]
            
            print('most_confused_index', most_confused_index)
        
            _, indices, _ = tss.find_index(most_confused_index, k=SAMPLE_PER_ITER) 
    
            print('len(indices)', len(indices))
            
            pool = np.delete(pool, indices)
    
            print('pool.shape', pool.shape)
            
            # Take things out of the space.
            tss.remove_from_space(indices)
            sampled_train_data_df = train_data_df.loc[indices,:]
            
            print("### ******** end ACLM split: {} ********".format(split))
            split += 1
            if split > max_iteration or pool.size == 0 or global_step >= args.max_steps:
                convergence_criterion_not_met = False
                
        checkpoint_path = save(
            model, optimizer, grad_scaler, scheduler, global_step, epoch, args
        )
        
        print("## ************ end ACLM epoch {} **********".format(epoch), )
        if global_step >= args.max_steps:
            break

    print("# **************** end main **********")
