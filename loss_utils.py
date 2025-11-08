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

import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils import mask_from_lens


class InstantSpeechLoss(nn.Module):
    def __init__(
        self, 
        dur_predictor_loss_scale=1.0,
        pitch_predictor_loss_scale=1.0, 
        attn_loss_scale=1.0,
        energy_predictor_loss_scale=0.1
    ):
        super(InstantSpeechLoss, self).__init__()
        self.duration_scale = dur_predictor_loss_scale
        self.pitch_scale = pitch_predictor_loss_scale
        self.energy_scale = energy_predictor_loss_scale
        self.attn_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self, model_outputs, batch):
        (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard, attn_hard_dur, attn_logprob), \
            (enc_hidden_outs, dec_hidden_outs, dur_hidden, pitch_hidden, energy_hidden) = model_outputs
        _, _, mel_tgt, _, _, _, _, in_lens, out_lens = batch

        dur_tgt = attn_hard_dur
        dur_lens = in_lens

        mel_tgt.requires_grad = False
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        log_dur_tgt = torch.log(dur_tgt.float() + 1)

        dur_pred_loss = F.mse_loss(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()
        
        ldiff = mel_tgt.size(1) - mel_out.size(1)
        mel_out = F.pad(mel_out, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = mel_tgt.ne(0).float()
        mel_loss = F.mse_loss(mel_out, mel_tgt, reduction='none')
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()
        
        if pitch_pred is not None:
            pitch_mask = dur_mask
            ldiff = pitch_tgt.size(2) - pitch_pred.size(2)
            pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0, 0, 0), value=0.0)
            pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
            pitch_loss = pitch_loss.squeeze(1)
            pitch_loss = (pitch_loss * pitch_mask).sum() / pitch_mask.sum()
        else:
            pitch_loss = 0
        
        if energy_pred is not None:
            energy_mask = dur_mask
            energy_pred = F.pad(energy_pred, (0, ldiff, 0, 0), value=0.0)
            energy_loss = F.mse_loss(energy_tgt, energy_pred, reduction='none')
            energy_loss = energy_loss.squeeze(1)
            energy_loss = (energy_loss * energy_mask).sum() / energy_mask.sum()
        else:
            energy_loss = 0

        attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)

        dur_pred_loss = dur_pred_loss * self.duration_scale
        pitch_loss = pitch_loss * self.pitch_scale
        energy_loss = energy_loss * self.energy_scale
        attn_loss = attn_loss * self.attn_scale

        total_loss = mel_loss + dur_pred_loss + pitch_loss + energy_loss + attn_loss

        stats = {}
        stats['total_loss'] = total_loss.detach()
        stats['mel_loss'] = mel_loss.detach()
        stats['dur_loss'] = dur_pred_loss.detach()
        stats['dur_error'] = (torch.abs(dur_pred - dur_tgt).sum() / dur_mask.sum()).detach()
        if pitch_pred is not None:
            stats['pitch_loss'] = pitch_loss.detach()
        if energy_pred is not None:
            stats['energy_loss'] = energy_loss.detach()
        stats['attn_loss'] = attn_loss.detach()

        return total_loss, stats
    
    
class InstantSpeechDistillationLoss(nn.Module):
    def __init__(self):
        super(InstantSpeechDistillationLoss, self).__init__()

    def forward(self, model_outputs, batch):
        student_outputs, teacher_outputs = model_outputs

        (stu_mel_out, stu_dec_mask, stu_dur_pred, stu_log_dur_pred, stu_pitch_pred, stu_energy_pred), \
            (stu_enc_hidden_outs, stu_dec_hidden_outs, stu_dur_hidden, stu_pitch_hidden, stu_energy_hidden) = student_outputs
        (tea_mel_out, tea_dec_mask, tea_dur_pred, tea_log_dur_pred, tea_pitch_pred, tea_energy_pred), \
            (tea_enc_hidden_outs, tea_dec_hidden_outs, tea_dur_hidden, tea_pitch_hidden, tea_energy_hidden) = teacher_outputs
        
        symbol_ids, num_symbols_of_each_word, speaker_id, in_lens = batch
                
        loss_fn = F.mse_loss
        
        dur_mask = mask_from_lens(in_lens)
        dur_distill_loss = F.mse_loss(stu_log_dur_pred, tea_log_dur_pred, reduction='none')
        dur_distill_loss = (dur_distill_loss * dur_mask).sum() / dur_mask.sum()
        
        pitch_mask = dur_mask
        pitch_distill_loss = F.mse_loss(stu_pitch_pred, tea_pitch_pred, reduction='none')
        pitch_distill_loss = pitch_distill_loss.squeeze(1)
        pitch_distill_loss = (pitch_distill_loss * pitch_mask).sum() / pitch_mask.sum()
        
        energy_mask = dur_mask
        energy_distill_loss = F.mse_loss(stu_energy_pred, tea_energy_pred, reduction='none')
        energy_distill_loss = energy_distill_loss.squeeze(1)
        energy_distill_loss = (energy_distill_loss * energy_mask).sum() / energy_mask.sum()
        
        
        num_mel_channels = tea_mel_out.size(2)
        mel_mask = tea_dec_mask.repeat(1, 1, num_mel_channels)
        mel_distill_loss = loss_fn(stu_mel_out, tea_mel_out, reduction='none')
        mel_distill_loss = (mel_distill_loss * mel_mask).sum() / mel_mask.sum()

        total_loss = mel_distill_loss + energy_distill_loss + pitch_distill_loss + dur_distill_loss

        stats = {}
        stats['total_loss'] = total_loss.detach()
        stats['mel_distill_loss'] = mel_distill_loss.detach()
        stats['pitch_distill_loss'] = pitch_distill_loss.detach()
        stats['dur_distill_loss'] = dur_distill_loss.detach()
        stats['energy_distill_loss'] = energy_distill_loss.detach()


        print(stats)

        return total_loss, stats


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_scores, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_scores.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_scores = attn_scores.squeeze(1)
        attn_scores = attn_scores.permute(1, 0, 2)

        # Add blank label
        attn_scores = F.pad(
            input=attn_scores,
            pad=(1, 0, 0, 0, 0, 0),
            value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(max_key_len+1, device=attn_scores.device, dtype=torch.long)
        attn_scores.masked_fill_(
            key_inds.view(1, 1, -1) > key_lens.view(1, -1, 1), # key_inds >= key_lens+1
            -1e15)
        attn_logprob = self.log_softmax(attn_scores)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.CTCLoss(
            attn_logprob, target_seqs,
            input_lengths=query_lens, target_lengths=key_lens)
        
        return cost