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

# Adapted from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch

from typing import Optional
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvReLUNorm, CausalConvReLUNorm, CausalConv1d
from common_utils import mask_from_lens
from alignment import b_mas, mas_width1
from attention import ConvAttention
from transformer import FFTransformer

import copy


def regulate_len(durations, enc_out, pace: float = 1.0,
                 mel_max_len: Optional[int] = None):
    """If target=None, then predicted durations are applied"""
    
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
        
    return enc_rep, dec_lens


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg


def get_lookbehind_num_symbols(word_index, num_symbols_of_each_word, lookbehind):
    num_words = len(num_symbols_of_each_word)
    num_lookbehind_symbols = 0
    lookbehind_word_start_index = word_index - lookbehind
    if lookbehind_word_start_index < 0:
        lookbehind_word_start_index = 0
    for index in range(lookbehind_word_start_index, word_index):
        num_lookbehind_symbols += num_symbols_of_each_word[index]
    return num_lookbehind_symbols


def get_lookahead_num_symbols(word_index, num_symbols_of_each_word, lookahead):
    num_words = len(num_symbols_of_each_word)
    num_lookahead_symbols = 0
    lookahead_word_end_index = word_index + 1 + lookahead
    if lookahead_word_end_index > num_words:
        lookahead_word_end_index = num_words
    for index in range(word_index + 1, lookahead_word_end_index):
        num_lookahead_symbols += num_symbols_of_each_word[index]
    return num_lookahead_symbols


def encoder_attention_mask(length, batch_num_symbols_of_each_word, lookbehind, lookahead, device):
    batch_size = len(batch_num_symbols_of_each_word)
    mask = torch.zeros((batch_size, length, length), device=device)
    for batch_index in range(batch_size):
        offset = 0
        num_words = len(batch_num_symbols_of_each_word[batch_index])
        for word_index in range(num_words):
            num_word_symbols = batch_num_symbols_of_each_word[batch_index][word_index]
            num_lookbehind_symbols = get_lookbehind_num_symbols(word_index, batch_num_symbols_of_each_word[batch_index], lookbehind)
            num_lookahead_symbols = get_lookahead_num_symbols(word_index, batch_num_symbols_of_each_word[batch_index], lookahead)
            mask[batch_index, offset: offset+num_word_symbols, offset-num_lookbehind_symbols: offset+num_word_symbols+num_lookahead_symbols] = 1
            offset += num_word_symbols
    return mask


def get_num_frames_of_each_word(num_symbols_of_each_word, num_frames_of_each_symbol):
    num_frames_of_each_word = []
    symbol_index = 0
    total_symbols = 0
    for num_symbols in num_symbols_of_each_word:
        total_symbols += num_symbols
        num_frames = []
        while symbol_index < total_symbols:
            num_frames.append(num_frames_of_each_symbol[symbol_index])
            symbol_index += 1
        num_frames = sum(num_frames)
        num_frames_of_each_word.append(num_frames)
    return num_frames_of_each_word


def get_lookbehind_num_frames(word_index, num_frames_of_each_word, lookbehind):
    num_words = len(num_frames_of_each_word)
    num_lookbehind_frames = 0
    lookbehind_word_start_index = word_index - lookbehind
    if lookbehind_word_start_index < 0:
        lookbehind_word_start_index = 0
    for index in range(lookbehind_word_start_index, word_index):
        num_lookbehind_frames += num_frames_of_each_word[index]
    return num_lookbehind_frames


def get_lookahead_num_frames(word_index, num_frames_of_each_word, lookahead):
    num_words = len(num_frames_of_each_word)
    num_lookahead_frames = 0
    lookahead_word_end_index = word_index + 1 + lookahead
    if lookahead_word_end_index > num_words:
        lookahead_word_end_index = num_words
    for index in range(word_index + 1, lookahead_word_end_index):
        num_lookahead_frames += num_frames_of_each_word[index]
    return num_lookahead_frames


def decoder_attention_mask(length, batch_num_symbols_of_each_word, batch_num_frames_of_each_symbol, lookbehind, lookahead, device):
    batch_size = len(batch_num_symbols_of_each_word)
    mask = torch.zeros((batch_size, length, length), device=device)
    for batch_index in range(batch_size):
        offset_symbol = 0
        offset_frame = 0
        num_words = len(batch_num_symbols_of_each_word[batch_index])
        num_frames_of_each_word = get_num_frames_of_each_word(batch_num_symbols_of_each_word[batch_index], batch_num_frames_of_each_symbol[batch_index])
        for word_index in range(num_words):
            num_word_symbols = batch_num_symbols_of_each_word[batch_index][word_index]
            word_num_frames_of_each_symbol = batch_num_frames_of_each_symbol[batch_index][offset_symbol: offset_symbol+num_word_symbols]
            offset_symbol += num_word_symbols
            num_word_frames = sum(word_num_frames_of_each_symbol)
            num_lookbehind_frames = get_lookbehind_num_frames(word_index, num_frames_of_each_word, lookbehind)
            num_lookahead_frames = get_lookahead_num_frames(word_index, num_frames_of_each_word, lookahead)
            mask[batch_index, offset_frame: offset_frame+num_word_frames, offset_frame-num_lookbehind_frames: offset_frame+num_word_frames+num_lookahead_frames] = 1
            offset_frame += num_word_frames
    return mask


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1, causal=False):
        super(TemporalPredictor, self).__init__()
        
        if causal:
            self.layers = nn.Sequential(*[
                CausalConvReLUNorm(input_size if i == 0 else filter_size, filter_size, 
                                   kernel_size=kernel_size, dropout=dropout)
                for i in range(n_layers)
            ])
        else:
            self.layers = nn.Sequential(*[
                ConvReLUNorm(input_size if i == 0 else filter_size, filter_size, 
                                   kernel_size=kernel_size, dropout=dropout)
                for i in range(n_layers)
            ])
        
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        hidden = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(hidden) * enc_out_mask
        return out, hidden


class InstantSpeech(nn.Module):

    def __init__(self, config: Config):
        super(InstantSpeech, self).__init__()

        self.config = config
        self.is_teacher = config.is_teacher
        self.lookahead = config.lookahead
        self.lookbehind = config.lookbehind

        self.is_causal = not self.is_teacher

        self.encoder = FFTransformer(
            n_layer=config.in_fft_n_layers, 
            n_head=config.in_fft_n_heads,
            d_model=config.symbol_embedding_dim,
            d_head=config.in_fft_d_head,
            d_inner=config.in_fft_conv1d_filter_size,
            kernel_size=config.in_fft_conv1d_kernel_size,
            dropout=config.p_in_fft_dropout,
            dropatt=config.p_in_fft_dropatt,
            dropemb=config.p_in_fft_dropemb,
            embed_input=True,
            d_embed=config.symbol_embedding_dim,
            n_embed=config.n_symbols,
            causal_pos_ff=self.is_causal,
            padding_idx=config.padding_idx)

        if config.n_speakers > 1:
            self.speaker_emb = nn.Embedding(config.n_speakers, config.symbol_embedding_dim)
        else:
            self.speaker_emb = None
        
        self.speaker_emb_weight = config.speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            input_size=config.in_fft_output_size,
            filter_size=config.dur_predictor_filter_size,
            kernel_size=config.dur_predictor_kernel_size,
            dropout=config.p_dur_predictor_dropout, 
            n_layers=config.dur_predictor_n_layers,
            causal=self.is_causal
        )

        self.decoder = FFTransformer(
            n_layer=config.out_fft_n_layers, 
            n_head=config.out_fft_n_heads,
            d_model=config.symbol_embedding_dim,
            d_head=config.out_fft_d_head,
            d_inner=config.out_fft_conv1d_filter_size,
            kernel_size=config.out_fft_conv1d_kernel_size,
            dropout=config.p_out_fft_dropout,
            dropatt=config.p_out_fft_dropatt,
            dropemb=config.p_out_fft_dropemb,
            embed_input=False,
            d_embed=config.symbol_embedding_dim,
            causal_pos_ff=self.is_causal
        )
        
        self.pitch_conditioning = config.pitch_conditioning

        if config.pitch_conditioning:
            self.pitch_predictor = TemporalPredictor(
                input_size=config.in_fft_output_size,
                filter_size=config.pitch_predictor_filter_size,
                kernel_size=config.pitch_predictor_kernel_size,
                dropout=config.p_pitch_predictor_dropout, 
                n_layers=config.pitch_predictor_n_layers,
                n_predictions=1,
                causal=self.is_causal
            )
            
            if self.is_causal:
                self.pitch_emb = CausalConv1d(
                    in_channels=1, 
                    out_channels=config.symbol_embedding_dim, 
                    kernel_size=config.pitch_embedding_kernel_size)
            else:
                self.pitch_emb = nn.Conv1d(
                    in_channels=1, 
                    out_channels=config.symbol_embedding_dim,
                    kernel_size=config.pitch_embedding_kernel_size,
                    padding=int((config.pitch_embedding_kernel_size - 1) / 2))

        self.energy_conditioning = config.energy_conditioning

        if config.energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                input_size=config.in_fft_output_size,
                filter_size=config.energy_predictor_filter_size,
                kernel_size=config.energy_predictor_kernel_size,
                dropout=config.p_energy_predictor_dropout,
                n_layers=config.energy_predictor_n_layers,
                n_predictions=1,
                causal=self.is_causal
            )
            
            if self.is_causal:
                self.energy_emb = CausalConv1d(
                    in_channels=1, 
                    out_channels=config.symbol_embedding_dim,
                    kernel_size=config.energy_embedding_kernel_size)
            else:
                self.energy_emb = nn.Conv1d(
                    in_channels=1, 
                    out_channels=config.symbol_embedding_dim,
                    kernel_size=config.energy_embedding_kernel_size,
                    padding=int((config.energy_embedding_kernel_size - 1) / 2))

        self.proj = nn.Linear(config.out_fft_output_size, config.n_mel_channels, bias=True)

        self.attention = ConvAttention(
            config.n_mel_channels,
            config.symbol_embedding_dim,
            use_query_proj=True
        )


    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(
                    attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())
        return attn_out
    

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(),
                             out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.get_device())


    def forward(
        self, 
        inputs, 
        use_gt_pitch=True, 
        use_gt_energy=True, 
        pace=1.0, 
        max_duration=75, 
        dur_guide=None, 
        pitch_guide=None, 
        energy_guide=None,
        text_only=False
    ):
        if self.is_teacher:
            return self.forward_teacher(
                inputs, 
                use_gt_pitch, 
                use_gt_energy, 
                pace, 
                max_duration,
                text_only
            )
        else:
            return self.forward_student(
                inputs, 
                use_gt_pitch, 
                use_gt_energy, 
                pace, 
                max_duration, 
                dur_guide, 
                pitch_guide, 
                energy_guide,
                text_only
            )
    

    def forward_teacher(
        self, 
        inputs, 
        use_gt_pitch, 
        use_gt_energy, 
        pace, 
        max_duration,
        text_only=False
    ):
        if text_only:
            (inputs, input_lens, num_symbols_of_each_word, speaker) = inputs
            mel_tgt, mel_lens, attn_prior = None, None, None
        else:
            (inputs, input_lens, num_symbols_of_each_word, mel_tgt, mel_lens, pitch_dense, energy_dense, attn_prior, speaker) = inputs
        
        mel_max_len = mel_tgt.size(2) if not text_only else None

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker)
            spk_emb.mul_(self.speaker_emb_weight)
        
        enc_out, enc_mask, enc_hidden_outs = self.encoder(inputs, conditioning=0)
        
        if self.speaker_emb is not None:
            enc_out += spk_emb
        
        if not text_only:
            text_emb = self.encoder.word_emb(inputs)
            attn_mask = mask_from_lens(input_lens)[..., None] == 0
            attn_soft, attn_logprob = self.attention(
                mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
                key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)
            attn_hard = self.binarize_attention_parallel(
                attn_soft, input_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            dur_tgt = attn_hard_dur
        else:
            attn_soft, attn_logprob, attn_hard, dur_tgt = None, None, None, None

        log_dur_pred, dur_hidden = self.duration_predictor(enc_out, enc_mask)
        log_dur_pred = log_dur_pred.squeeze(-1)

        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
        
        if self.pitch_conditioning:
            pitch_pred, pitch_hidden = self.pitch_predictor(enc_out, enc_mask)
            pitch_pred = pitch_pred.permute(0, 2, 1)
            if not text_only:
                pitch_tgt = average_pitch(pitch_dense, dur_tgt)
            else:
                pitch_tgt = None
            if use_gt_pitch and pitch_tgt is not None:
                pitch_emb = self.pitch_emb(pitch_tgt)
            else:
                pitch_emb = self.pitch_emb(pitch_pred)
            enc_out = enc_out + pitch_emb.transpose(1, 2)
        else:
            pitch_pred = None
            pitch_tgt = None
            pitch_hidden = None

        if self.energy_conditioning:
            energy_pred, energy_hidden = self.energy_predictor(enc_out, enc_mask)
            energy_pred = energy_pred.permute(0, 2, 1)
            if not text_only:
                energy_tgt = average_pitch(energy_dense, dur_tgt)
                energy_tgt = torch.log(1.0 + energy_tgt)
            else:
                energy_tgt = None
            if use_gt_energy and energy_tgt is not None:
                energy_emb = self.energy_emb(energy_tgt)
            else:
                energy_emb = self.energy_emb(energy_pred)
            enc_out = enc_out + energy_emb.transpose(1, 2) 
        else:
            energy_pred = None
            energy_tgt = None
            energy_hidden = None
        
        if text_only or dur_tgt is None:
            len_regulated, dec_lens = regulate_len(dur_pred, enc_out, pace, mel_max_len)
        else:
            len_regulated, dec_lens = regulate_len(dur_tgt, enc_out, pace, mel_max_len)
        
        dec_out, dec_mask, dec_hidden_outs = self.decoder(len_regulated, dec_lens)
        
        mel_out = self.proj(dec_out)
        
        return (
            (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard, attn_hard_dur, attn_logprob) if not text_only 
            else (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, energy_pred)
        ), (enc_hidden_outs, dec_hidden_outs, dur_hidden, pitch_hidden, energy_hidden)
    

    def forward_student(
        self, 
        inputs, 
        use_gt_pitch=False, 
        use_gt_energy=False, 
        pace=1.0, 
        max_duration=75,
        dur_guide=None, 
        pitch_guide=None, 
        energy_guide=None,
        text_only=False
    ):
        if text_only:
            (inputs, input_lens, num_symbols_of_each_word, speaker) = inputs
            mel_tgt, mel_lens, attn_prior = None, None, None
        else:
            (inputs, input_lens, num_symbols_of_each_word, mel_tgt, mel_lens, pitch_dense, energy_dense, attn_prior, speaker) = inputs

        mel_max_len = mel_tgt.size(2) if not text_only else None

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker)
            spk_emb.mul_(self.speaker_emb_weight)

        inp_max_len = inputs.size(1)
        enc_attn_mask = encoder_attention_mask(
            length=inp_max_len,
            batch_num_symbols_of_each_word=num_symbols_of_each_word,
            lookbehind=self.lookbehind,
            lookahead=self.lookahead,
            device=inputs.device
        )
        enc_out, enc_mask, enc_hidden_outs = self.encoder(inputs, conditioning=0, aux_attn_mask=enc_attn_mask)
        if self.speaker_emb is not None:
            enc_out += spk_emb

        if not text_only:
            text_emb = self.encoder.word_emb(inputs)
            attn_mask = mask_from_lens(input_lens)[..., None] == 0
            attn_soft, attn_logprob = self.attention(
                mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
                key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)
            attn_hard = self.binarize_attention_parallel(
                attn_soft, input_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            dur_tgt = attn_hard_dur
        else:
            attn_soft, attn_logprob, attn_hard, dur_tgt = None, None, None, None

        log_dur_pred, dur_hidden = self.duration_predictor(enc_out, enc_mask)
        log_dur_pred = log_dur_pred.squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        if self.pitch_conditioning:
            pitch_pred, pitch_hidden = self.pitch_predictor(enc_out, enc_mask)
            pitch_pred = pitch_pred.permute(0, 2, 1)
            if not text_only:
                pitch_tgt = average_pitch(pitch_dense, dur_tgt)
            else:
                pitch_tgt = None
            if pitch_guide is not None:
                pitch_emb = self.pitch_emb(pitch_guide)
            elif use_gt_pitch and pitch_tgt is not None:
                pitch_emb = self.pitch_emb(pitch_tgt)
            else:
                pitch_emb = self.pitch_emb(pitch_pred)
            enc_out += pitch_emb.transpose(1, 2)
        else:
            pitch_pred, pitch_tgt, pitch_hidden = None, None, None

        if self.energy_conditioning:
            energy_pred, energy_hidden = self.energy_predictor(enc_out, enc_mask)
            energy_pred = energy_pred.permute(0, 2, 1)
            if not text_only:
                energy_tgt = average_pitch(energy_dense, dur_tgt)
                energy_tgt = torch.log(1.0 + energy_tgt)
            else:
                energy_tgt = None
            if energy_guide is not None:
                energy_emb = self.energy_emb(energy_guide)
            elif use_gt_energy and energy_tgt is not None:
                energy_emb = self.energy_emb(energy_tgt)
            else:
                energy_emb = self.energy_emb(energy_pred)
            enc_out += energy_emb.transpose(1, 2)
        else:
            energy_pred, energy_tgt, energy_hidden = None, None, None

        if dur_guide is not None:
            len_regulated, dec_lens = regulate_len(dur_guide, enc_out, pace, mel_max_len)
            num_frames_of_each_symbol = dur_guide.float() / pace + 0.5
        elif dur_tgt is not None:
            len_regulated, dec_lens = regulate_len(dur_tgt, enc_out, pace, mel_max_len)
            num_frames_of_each_symbol = dur_tgt.float() / pace + 0.5
        else:
            len_regulated, dec_lens = regulate_len(dur_pred, enc_out, pace, mel_max_len)
            num_frames_of_each_symbol = dur_pred.float() / pace + 0.5
        
        num_frames_of_each_symbol = num_frames_of_each_symbol.int().tolist()

        if mel_max_len is None:
            mel_max_len = len_regulated.size(1)

        dec_attn_mask = decoder_attention_mask(
            length=mel_max_len,
            batch_num_symbols_of_each_word=num_symbols_of_each_word,
            batch_num_frames_of_each_symbol=num_frames_of_each_symbol,
            lookbehind=self.lookbehind,
            lookahead=self.lookahead,
            device=len_regulated.device
        )
        
        dec_out, dec_mask, dec_hidden_outs = self.decoder(len_regulated, dec_lens, aux_attn_mask=dec_attn_mask)
        mel_out = self.proj(dec_out)

        return (
            (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard, attn_hard_dur, attn_logprob) if not text_only 
            else (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, energy_pred)
        ), (enc_hidden_outs, dec_hidden_outs, dur_hidden, pitch_hidden, energy_hidden)


    def infer(
        self, 
        inputs, 
        pace=1.0, 
        max_duration=75, 
        speaker=0, 
        num_symbols_of_each_word=None, 
        past=None, 
        final=False
    ):
        if self.is_teacher:
            return self.infer_teacher(
                inputs, 
                pace, 
                max_duration, 
                speaker
            )
        else:
            return self.infer_student(
                inputs, 
                pace, 
                max_duration, 
                speaker, 
                num_symbols_of_each_word, 
                past, 
                final
            )
        
    def infer_teacher(
        self, 
        inputs, 
        pace=1.0, 
        max_duration=75, 
        speaker=0
    ):
        if self.speaker_emb is None:
            spk_emb = 0

        elif isinstance(speaker, int):
            speaker = (torch.ones(inputs.size(0)).long().to(inputs.device) * speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
        else:
            spk_emb = self.speaker_emb(speaker)
            spk_emb.mul_(self.speaker_emb_weight)
        
        enc_out, enc_mask, enc_hidden_outs = self.encoder(inputs, conditioning=0)
        
        log_dur_pred, dur_hidden = self.duration_predictor((enc_out + spk_emb), enc_mask)
        log_dur_pred = log_dur_pred.squeeze(-1)
        
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
        
        if self.pitch_conditioning:
            pitch_pred, pitch_hidden = self.pitch_predictor((enc_out + spk_emb), enc_mask)
            pitch_pred = pitch_pred.permute(0, 2, 1)
        
            pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
            enc_out = enc_out + pitch_emb
        else:
            pitch_pred = None
            pitch_hidden = None
        
        if self.energy_conditioning:
            energy_pred, energy_hidden = self.energy_predictor((enc_out + spk_emb), enc_mask)
            energy_pred = energy_pred.permute(0, 2, 1)
            
            energy_emb = self.energy_emb(energy_pred).transpose(1, 2)
            enc_out = enc_out + energy_emb
        else:
            energy_pred = None
            energy_hidden = None
            
        len_regulated, dec_lens = regulate_len(dur_pred, (enc_out + spk_emb), pace, mel_max_len=None)
        
        dec_out, dec_mask, dec_hidden_outs = self.decoder(len_regulated, dec_lens)
            
        mel_out = self.proj(dec_out)
        
        return (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, energy_pred), \
            (enc_hidden_outs, dec_hidden_outs, dur_hidden, pitch_hidden, energy_hidden)
        
    
    def infer_student(
        self, 
        inputs, 
        pace=1.0, 
        max_duration=75, 
        speaker=0, 
        num_symbols_of_each_word=None, 
        past=None, 
        final=False
    ):
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = (torch.ones(inputs.size(0)).long().to(inputs.device) * speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
        
        past_inputs = past["inputs"]
        past_inputs_length = past_inputs.size(1)
        new_inputs_length = inputs.size(1)
        inputs = torch.concat((past_inputs, inputs), dim=1)
        
        past_num_symbols_of_each_word = past["num_symbols_of_each_word"]
        new_num_symbols_of_each_word = copy.deepcopy(past_num_symbols_of_each_word)
        past_num_symbols_of_each_word[0].append(num_symbols_of_each_word[0][0])
        new_num_symbols_of_each_word[0] += num_symbols_of_each_word[0]
        num_symbols_of_each_word = new_num_symbols_of_each_word
        
        inp_max_len = inputs.size(1)

        enc_attn_mask = encoder_attention_mask(
            length=inp_max_len,
            batch_num_symbols_of_each_word=num_symbols_of_each_word,
            lookbehind=self.lookbehind,
            lookahead=self.lookahead,
            device=inputs.device
        )

        enc_out, enc_mask, enc_hidden_outs = self.encoder(inputs, conditioning=0, aux_attn_mask=enc_attn_mask)
        
        log_dur_pred, dur_hidden = self.duration_predictor((enc_out + spk_emb), enc_mask)
        log_dur_pred = log_dur_pred.squeeze(-1)
        
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
        new_inputs_dur_pred = dur_pred[:, -new_inputs_length:]
        new_inputs_num_frames = torch.sum((new_inputs_dur_pred / pace + 0.5).int(), dim=1)
        
        if self.lookahead > 0:
            num_symbols_to_cut = sum(num_symbols_of_each_word[0][-self.lookahead:])
            dur_of_symbols_to_cut = dur_pred[:, -num_symbols_to_cut:]
            num_frames_to_cut = torch.sum((dur_of_symbols_to_cut / pace + 0.5).int(), dim=1)
        else:
            num_symbols_to_cut = 0
            dur_of_symbols_to_cut = dur_pred
            num_frames_to_cut = 0
        
        if self.pitch_conditioning:
            pitch_pred, pitch_hidden = self.pitch_predictor((enc_out + spk_emb), enc_mask)
            pitch_pred = pitch_pred.permute(0, 2, 1)
            pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
            enc_out = enc_out + pitch_emb
        else:
            pitch_pred = None
            pitch_hidden = None
        
        if self.energy_conditioning:
            energy_pred, energy_hidden = self.energy_predictor((enc_out + spk_emb), enc_mask)
            energy_pred = energy_pred.permute(0, 2, 1)
            energy_emb = self.energy_emb(energy_pred).transpose(1, 2)
            enc_out = enc_out + energy_emb
        else:
            energy_pred = None
            energy_hidden = None
            
        len_regulated, dec_lens = regulate_len(dur_pred, (enc_out + spk_emb), pace, mel_max_len=None)
        
        mel_max_len = len_regulated.size(1)
        num_frames_of_each_symbol = (dur_pred / pace + 0.5).int().tolist()
        dec_attn_mask = decoder_attention_mask(
            length=mel_max_len,
            batch_num_symbols_of_each_word=num_symbols_of_each_word,
            batch_num_frames_of_each_symbol=num_frames_of_each_symbol,
            lookbehind=self.lookbehind,
            lookahead=self.lookahead,
            device=len_regulated.device)
        
        dec_out, dec_mask, dec_hidden_outs = self.decoder(len_regulated, dec_lens, aux_attn_mask=dec_attn_mask)
            
        mel_out = self.proj(dec_out).permute(0, 2, 1)
        
        mel_out_increment = mel_out[:, :, -new_inputs_num_frames:]
        
        if not final and num_frames_to_cut > 0:
            mel_out_increment = mel_out_increment[:, :, :-num_frames_to_cut]
        
        if num_symbols_to_cut != 0:
            past["inputs"] = inputs[:, :-num_symbols_to_cut]
        else:
            past["inputs"] = inputs
            
        past["num_symbols_of_each_word"] = past_num_symbols_of_each_word
        
        return mel_out_increment, mel_out, past