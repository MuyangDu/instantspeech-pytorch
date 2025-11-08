
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

from hydra.utils import to_absolute_path
from accelerate.utils import set_seed
from omegaconf import open_dict
from logging_utils import Logger
import torch
from einops import rearrange
from torch.special import gammaln
import numpy as np

    
def mask_from_lens(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def check_args_and_env(args):
    assert args.optim.batch_size % args.optim.grad_acc == 0
    assert args.eval.every_steps % args.logging.every_steps == 0

    if args.device == 'gpu':
        assert torch.cuda.is_available(), 'We use GPU to train/eval the model'

    assert not (args.eval_only and args.predict_only)


def opti_flags(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == 'bf16' and args.device == 'gpu':
        args.model.add_config.is_bf16 = True


def setup_basics(accelerator, args):
    check_args_and_env(args)
    update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    logger = Logger(args=args, accelerator=accelerator)

    return logger


def update_paths(args):
    args.model.config_path = to_absolute_path(args.model.config_path)
    args.data.train_filelist_path = to_absolute_path(args.data.train_filelist_path)
    args.data.dev_filelist_path = to_absolute_path(args.data.dev_filelist_path)
    args.data.pitch_mean_path = to_absolute_path(args.data.pitch_mean_path)
    args.data.pitch_std_path = to_absolute_path(args.data.pitch_std_path)
    if args.model.checkpoint_path != "":
        args.model.checkpoint_path = to_absolute_path(args.model.checkpoint_path)
    if args.model.restore_from != "":
        args.model.restore_from = to_absolute_path(args.model.restore_from)
    if args.model.teacher_checkpoint_path != "":
        args.model.teacher_checkpoint_path = to_absolute_path(args.model.teacher_checkpoint_path)
    if args.vocoder_model.ckpt_path != "":
        args.vocoder_model.ckpt_path = to_absolute_path(args.vocoder_model.ckpt_path)

def logbeta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logcombinations(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logbetabinom(n, a, b, x):
    return logcombinations(n, x) + logbeta(x + a, n - x + b) - logbeta(a, b)


def get_beta_binomial_prior(encoder_length: int, decoder_length: int, scaling_factor: float = 1.0) -> np.array:
    x = rearrange(torch.arange(0, encoder_length), "b -> 1 b")
    y = rearrange(torch.arange(1, decoder_length + 1), "b -> b 1")
    a = scaling_factor * y
    b = scaling_factor * (decoder_length + 1 - y)
    n = torch.FloatTensor([encoder_length - 1])
    beta_binomial_prior = logbetabinom(n, a, b, x).exp()

    return beta_binomial_prior


def attention_prior_annealing(attention_prior, current_train_step, start_scale_down_step, end_step):
    if current_train_step >= end_step:
        return None
    elif current_train_step > start_scale_down_step and current_train_step < end_step:
        total_annealing_steps = end_step - start_scale_down_step
        curr_annealing_step = current_train_step - start_scale_down_step
        attention_prior = attention_prior + ((1.0 - attention_prior) * curr_annealing_step / total_annealing_steps)
        return attention_prior
    else:
        return attention_prior


def g2p(word, lookbehind_words, lookahead_words, g2p_backend):
    def phonemize_and_split(words):
        result = g2p_backend.phonemize([" ".join(words)], strip=True)
        return result[0].split(" ") if result else []
    full_context = lookbehind_words + [word] + lookahead_words
    phonemes = phonemize_and_split(full_context)
    if len(phonemes) == len(full_context):
        current = phonemes[len(lookbehind_words)]
        lookahead = phonemes[len(lookbehind_words) + 1:]
        return current, lookahead
    forward_context = [word] + lookahead_words
    phonemes = phonemize_and_split(forward_context)
    if len(phonemes) == len(forward_context):
        current = phonemes[0]
        lookahead = phonemes[1:]
        return current, lookahead
    backward_context = lookbehind_words + [word]
    phonemes = phonemize_and_split(backward_context)
    if len(phonemes) == len(backward_context):
        current = phonemes[-1]
        lookahead_phonemes = phonemize_and_split(lookahead_words)
        return current, lookahead_phonemes or [""]
    phonemes = g2p_backend.phonemize([word], strip=True)
    current = phonemes[0] if phonemes else ""
    lookahead_phonemes = phonemize_and_split(lookahead_words)
    return current, lookahead_phonemes or [""]


class CrossFade(torch.nn.Module):
    
    def __init__(self, num_overlap_samples, device="cuda"):
        super(CrossFade, self).__init__()
        self.num_overlap_samples = num_overlap_samples
        self.fade_in_coeff, self.fade_out_coeff = self.get_crossfade_coeff()
        self.fade_in_coeff = self.fade_in_coeff.float().to(device)
        self.fade_out_coeff = self.fade_out_coeff.float().to(device)
    
    def get_crossfade_coeff(self):
        fade_len = self.num_overlap_samples
        hann_win = np.hanning(fade_len * 2)
        fade_in = hann_win[:fade_len]
        fade_out = hann_win[fade_len:]
        fade_out_coeff = torch.tensor(fade_out)
        fade_in_coeff = torch.tensor(fade_in)

        return fade_in_coeff, fade_out_coeff
    
    def apply_fade_out(self, waveform):
        waveform_tail = waveform[:, -self.num_overlap_samples:]
        waveform_tail = waveform_tail * self.fade_out_coeff
        waveform = torch.cat((waveform[:, :-self.num_overlap_samples], waveform_tail), dim=1)
        
        return waveform
    
    def apply_fade_in(self, waveform):
        waveform_head = waveform[:, :self.num_overlap_samples]
        waveform_head = waveform_head * self.fade_in_coeff
        waveform = torch.cat((waveform_head, waveform[:, self.num_overlap_samples:]), dim=1)
        
        return waveform
    
    def cross_fade(self, waveform_chunk_to_respond, pre_waveform_chunk_tail):
        waveform_chunk_to_respond = self.apply_fade_in(waveform_chunk_to_respond)
        pre_waveform_chunk_tail = self.apply_fade_out(pre_waveform_chunk_tail)
        waveform_chunk_to_respond_head = waveform_chunk_to_respond[:, :self.num_overlap_samples]
        waveform_chunk_to_respond_head = waveform_chunk_to_respond_head + pre_waveform_chunk_tail[:, -self.num_overlap_samples:]
        waveform_chunk_to_respond = torch.cat((
            waveform_chunk_to_respond_head, waveform_chunk_to_respond[:, self.num_overlap_samples:]), dim=1)
        
        return waveform_chunk_to_respond
    
    def forward(self, waveform_chunk, pre_waveform_chunk_tail):
        cur_waveform_chunk_tail = waveform_chunk[:, -self.num_overlap_samples:]
        waveform_chunk_to_respond = waveform_chunk[:, :-self.num_overlap_samples]
        waveform_chunk_to_respond = self.cross_fade(waveform_chunk_to_respond, pre_waveform_chunk_tail)
            
        return waveform_chunk_to_respond, cur_waveform_chunk_tail