# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

from functools import partial
import logging
import os
from pathlib import Path
import socket

import torch
import wandb

from dist import get_world_size, setup_torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

log = logging.getLogger(__name__)

WBRUN = None
# if get_rank() == 0:

PORT = os.environ.get('MASTER_PORT', '5432')
RANK, WORLD_SIZE = setup_torch('DDP', port=PORT)
HERE = Path(os.path.abspath(__file__)).parent

ENV_FILTERS = [
    'PS1',
    'LSCOLORS',
    'LS_COLORS',
]

ENV_PREFIXES = [
    '_ModuleTable',
    'BASH_FUNC_',
]

if RANK == 0:
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR', None)
    if tensorboard_dir is not None:
        log.info(f'Patching tensorboard from {tensorboard_dir}')
        wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    # os.environ['WANDB_RUN_GROUP'] = f'experiment-{generate_id()}'
    WBRUN = wandb.init(
        project='Megatron-LM-Nvidia',
        sync_tensorboard=True,
        dir=tensorboard_dir,
        resume='allow',
        # dir=os.getcwd(),
        # sync_tensorboard=True,
        # group=f'experiment-{generate_id()}'
    )
    assert WBRUN is not None and WBRUN is wandb.run
    wandb.run.log_code(HERE.as_posix())  # type:ignore
    # WBRUN.config.update(args)
    # WBRUN.log_code(HERE.as_posix())
    model_size = os.environ.get('MODEL_SIZE', None)
    if model_size is not None:
        WBRUN.config.update({'MODEL_SIZE': model_size})
    if WBRUN is not None:
        assert WBRUN is wandb.run
        WBRUN.config.update({'world_size': get_world_size()})
        # env = dict(os.environ)
        # _ = env.pop('LS_COLORS', None)
        # WBRUN.config.update({'env': env})
        env = dict(os.environ)
        for key in ENV_FILTERS + ['LS_COLORS', 'LSCOLORS']:
            _ = env.pop(key, None)
        WBRUN.config.update({'env': env})
        hostname = socket.gethostbyaddr(socket.gethostname())[0]
        if hostname.startswith('theta'):
            WBRUN.config.update({'machine': 'ThetaGPU'})
        elif hostname.startswith('x3'):
            WBRUN.config.update({'machine': 'Polaris'})
        elif hostname.startswith('x1'):
            WBRUN.config.update({'machine': 'Sunspot'})
        else:
            WBRUN.config.update({'machine': hostname})

# RANK, WORLD_SIZE = setup_torch(
#     backend='DDP',
#     port='5432',
# )
# log.info(f'Hello from {RANK} / {WORLD_SIZE}')

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    if wandb.run is not None:
        wandb.run.config.update(config)
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
