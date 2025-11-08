# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import torch
from model_vocoder import Generator
from model import InstantSpeech
from data_utils import (
    InstantSpeechDataset,
    InstantSpeechCollate,
    InstantSpeechTextDataset,
    InstantSpeechTextCollate
)
from loss_utils import (
    InstantSpeechLoss,
    InstantSpeechDistillationLoss
)
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from omegaconf import open_dict
from config import Config
from types import SimpleNamespace

def get_config(args):
    config = Config.from_json(args.model.config_path)

    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f'config does not have attribute {k}'
            setattr(config, k, v)

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f'config already has attribute {k}'
            setattr(config, k, v)

    return config


def get_model(args, config):
    model = InstantSpeech(config)

    if args.model.checkpoint_path and args.model.checkpoint_path.strip() != "":
        state_dict = load_file(args.model.checkpoint_path.strip())
        model.load_state_dict(state_dict)

    with open_dict(args):
        total_params= get_model_params(model)
        args.total_params = total_params

    if args.mode == "distill":
        # teacher model must have the same config as the student model
        config_teacher = copy.deepcopy(config)
        config_teacher.is_teacher = True
        teacher_model = InstantSpeech(config_teacher)
        if args.model.teacher_checkpoint_path and args.model.teacher_checkpoint_path.strip() != "":
            teacher_state_dict = load_file(args.model.teacher_checkpoint_path.strip())
            teacher_model.load_state_dict(teacher_state_dict)
        else:
            raise ValueError("teacher_checkpoint_path is required for distillation.")
    else:
        teacher_model = None
        
    return model, teacher_model


def get_model_params(model):
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.nelement()
        
    return total_params


def get_vocoder_model(args, config):
    if args.vocoder_model.klass == "causal_hifigan":
        model = Generator(config)
        if os.path.exists(args.vocoder_model.ckpt_path):
            model.load_state_dict(torch.load(args.vocoder_model.ckpt_path, map_location="cpu"))
        else:
            raise FileNotFoundError(
            f"Pre-trained vocoder checkpoint '{args.vocoder_model.ckpt_path}' not found. "
            f"Please refer to the README or the access form to obtain a checkpoint."
        )
        model.remove_weight_norm()
        vocoder_hparams = SimpleNamespace(name="causal_hifigan", sampling_rate=config.sample_rate, hop_size=config.stft_hop_length)
    else:
        raise NotImplementedError

    model = model.eval()

    return model, vocoder_hparams


def get_loss_function(args):
    if args.mode == "pretrain":
        loss_function = InstantSpeechLoss(attn_loss_scale=0.1)
    elif args.mode == "distill":
        loss_function = InstantSpeechDistillationLoss()
    else:
        raise NotImplementedError
    return loss_function


def load_dataset_splits(args, config):
    if args.mode == "pretrain":
        train_dataset = InstantSpeechDataset(
            filelist_path=args.data.train_filelist_path, 
            pitch_mean_path=args.data.pitch_mean_path, 
            pitch_std_path=args.data.pitch_std_path, 
            num_speakers=config.n_speakers
        )
        dev_dataset = InstantSpeechDataset(
            filelist_path=args.data.dev_filelist_path, 
            pitch_mean_path=args.data.pitch_mean_path, 
            pitch_std_path=args.data.pitch_std_path, 
            num_speakers=config.n_speakers
        )
    elif args.mode == "distill":
        train_dataset = InstantSpeechTextDataset(
            filelist_path=args.data.train_filelist_path, 
            num_speakers=config.n_speakers
        )
        dev_dataset = InstantSpeechTextDataset(
            filelist_path=args.data.dev_filelist_path, 
            num_speakers=config.n_speakers
        )
    else:
        raise NotImplementedError

    return train_dataset, dev_dataset


def get_data_collator(args):
    if args.mode == "pretrain":
        data_collator = InstantSpeechCollate()
    elif args.mode == "distill":
        data_collator = InstantSpeechTextCollate()
    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(args, config):
    train_dataset, dev_dataset = load_dataset_splits(args, config)
    data_collator = get_data_collator(args)
    batch_size = args.optim.batch_size // args.optim.grad_acc

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with open_dict(args):
        args.data.train_batches = len(train_dataloader)
        args.data.dev_batches = len(dev_dataloader)
        if args.optim.epochs > 0:
            args.optim.total_steps = (len(train_dataloader) // args.optim.grad_acc) * args.optim.epochs 

    return train_dataloader, dev_dataloader


def get_optimizer(model, args):
    if args.optim.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.optim.base_lr, 
            weight_decay=args.optim.weight_decay
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args):
    if args.optim.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )
    else:
        raise NotImplementedError

    return lr_scheduler