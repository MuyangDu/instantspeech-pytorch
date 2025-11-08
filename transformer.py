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

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CausalConv1d

from common_utils import mask_from_lens


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1),
                                    torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]
        
    def export_forward(self, pos_seq):
        pos_seq = torch.unsqueeze(pos_seq, -1)
        invfreq = torch.unsqueeze(self.inv_freq, 0)
        sinusoid_inp = torch.matmul(pos_seq, invfreq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        
        return pos_emb


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False, causal_pos_ff=False):
        super(PositionwiseConvFF, self).__init__()
        
        self.causal_pos_ff = causal_pos_ff
        
        self.kernel_size = kernel_size

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        
        
        if causal_pos_ff:
            self.CoreNet = nn.Sequential(
                CausalConv1d(d_model, d_inner, kernel_size, stride=1),
                nn.ReLU(),
                # nn.Dropout(dropout),  # worse convergence
                CausalConv1d(d_inner, d_model, kernel_size, stride=1),
                nn.Dropout(dropout),
            )
        else:
            self.CoreNet = nn.Sequential(
                nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
                nn.ReLU(),
                # nn.Dropout(dropout),  # worse convergence
                nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
                nn.Dropout(dropout),
            )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp, past_state=None):
        return self._forward(inp, past_state)

    def _forward(self, inp, past_state):
        if past_state is not None:
            new_past_state = []
            
        core_out = inp.transpose(1, 2)
        
        if self.pre_lnorm:
            core_out = self.layer_norm(core_out).to(inp.dtype)
            
        if past_state is not None:
            new_past_state.append(core_out[:, :, -self.kernel_size:])
            core_out = torch.cat((past_state[0], core_out), dim=2)
        
        core_out = self.CoreNet[0](core_out) # first causal conv
        
        core_out = self.CoreNet[1](core_out) # relu
        
        if past_state is not None:
            core_out = core_out[:, :, -inp.size(1):]
            new_past_state.append(core_out[:, :, -self.kernel_size:])
            core_out = torch.cat((past_state[1], core_out), dim=2)
        
        core_out = self.CoreNet[2](core_out) # second causal conv
        
        if past_state is not None:
            core_out = core_out[:, :, -inp.size(1):]
        
        core_out = self.CoreNet[3](core_out) # dropout
        
        core_out = core_out.transpose(1, 2)
        
        output = inp + core_out
        
        if not self.pre_lnorm:
             output = self.layer_norm(output).to(inp.dtype)
        
        if past_state is not None:
            return output, new_past_state
        else:
            return output
    
    
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.pre_lnorm = pre_lnorm
        self.scale = 1 / (d_head ** 0.5)
        self.all_head_size = n_head * d_head
        
        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.dropout_output = nn.Dropout(dropout)
        self.dropout_attention = nn.Dropout(dropatt)
        
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        

    def forward(self, inp, attn_mask=None, past_key_value=None, aux_attn_mask=None):
        return self._forward(inp, attn_mask, past_key_value, aux_attn_mask)

    def _forward(self, inp, attn_mask=None, past_key_value=None, aux_attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head
        
        head_q = self.query(inp)
        head_k = self.key(inp)
        head_v = self.value(inp)
        
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)
        
        if past_key_value is not None:
            past_key = past_key_value[0]
            past_value = past_key_value[1]
            
            head_k = torch.cat((past_key, head_k), dim=1)
            head_v = torch.cat((past_value, head_v), dim=1)
            
            new_past_key_value = [head_k, head_v]
            new_past_key_value = torch.stack(new_past_key_value)
            
            q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
            
            k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1) + past_key.size(1), d_head)
            v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1) + past_value.size(1), d_head)
        else:
            q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
            k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
            v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))

        attn_score.mul_(self.scale)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).to(attn_score.dtype)
            
            if past_key_value is not None:
                attn_mask = attn_mask.repeat(n_head, inp.size(1), 1).int() * -1000.0
                past_attn_mask = torch.zeros(
                    (attn_mask.size(0), attn_mask.size(1), past_value.size(1)), 
                    device=attn_mask.device,
                    dtype=attn_mask.dtype)

                attn_mask = torch.cat((past_attn_mask, attn_mask), dim=2)
            else:
                attn_mask = attn_mask.repeat(1, n_head, attn_mask.size(3), 1)

            if aux_attn_mask is not None:
                
                aux_attn_mask = aux_attn_mask.unsqueeze(1)
                aux_attn_mask = aux_attn_mask.repeat(1, n_head, 1, 1)
                aux_attn_mask = (1 - aux_attn_mask)
                attn_mask += aux_attn_mask
            
            attn_mask = attn_mask.bool()

            attn_mask = attn_mask.reshape(attn_mask.shape[0] * attn_mask.shape[1], attn_mask.shape[2], attn_mask.shape[3])
            attn_score = attn_score.masked_fill_(attn_mask, -1e5)

        attn_prob = F.softmax(attn_score, dim=2)
        
        attn_prob = self.dropout_attention(attn_prob)
        
        attn_vec = torch.bmm(attn_prob, v)
        
        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout_output(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)
        
        output = output.to(attn_out.dtype)
        
        if past_key_value is not None:
            return output, new_past_key_value
        else:
            return output
        

class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, causal_pos_ff,
                 **kwargs):
        super(TransformerLayer, self).__init__()
        
        self.pos_ff_kernel_size = kernel_size
        self.causal_pos_ff = causal_pos_ff

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout, 
                                         pre_lnorm=kwargs.get('pre_lnorm'), causal_pos_ff=causal_pos_ff)

    
    def forward(self, dec_inp, mask=None, past_key_value=None, past_state=None, aux_attn_mask=None):
        if past_key_value is not None:
            output, new_past_key_value = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2), past_key_value=past_key_value, aux_attn_mask=aux_attn_mask)
        else:
            output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2), aux_attn_mask=aux_attn_mask)
        
        output *= mask
        
        if past_state is not None:
            output, new_past_state = self.pos_ff(output, past_state)
        else:
            output = self.pos_ff(output)
        
        output *= mask
        
        if past_key_value is not None and past_state is not None:
            return output, new_past_key_value, new_past_state
        elif past_key_value is not None:
            return output, new_past_key_value
        elif past_state is not None:
            return output, new_past_state
        else:
            return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False, causal_pos_ff=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx
        
        self.n_layers = n_layer
        self.causal_pos_ff = causal_pos_ff
        self.pos_conv_ff_kernel_size = kernel_size
        self.d_inner = d_inner

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model,
                                         padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm, causal_pos_ff=causal_pos_ff)
            )


    def forward(self, dec_inp, seq_lens=None, conditioning=0, aux_attn_mask=None):
        if self.word_emb is None:
            inp = dec_inp
            mask = mask_from_lens(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != self.padding_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        
        out = self.drop(inp + pos_emb + conditioning)
        
        hidden_outs = []
        
        for layer in self.layers:
            out = layer(out, mask=mask, aux_attn_mask=aux_attn_mask)
            hidden_outs.append(out)

        return out, mask, hidden_outs