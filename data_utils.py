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
from text_utils import (
    symbols_to_ids,
    add_bos_and_eos_ids
)
from common_utils import get_beta_binomial_prior
import numpy as np
import torch
import random
import json


class InstantSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, filelist_path, pitch_mean_path, pitch_std_path, num_speakers):
        self.metadata = self.load_filelist(filelist_path)
        self.pitch_mean = np.load(pitch_mean_path)
        self.pitch_std = np.load(pitch_std_path)
        self.num_speakers = num_speakers

    def load_filelist(self, path):
        metadata = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                metadata.append(json.loads(line.strip()))
        return metadata
    
    def get_item(self, meta):
        symbols = meta["symbols"]
        mel_path = meta["mel"]
        pitch_path = meta["pitch"]
        speaker = meta["speaker"]
        symbols = symbols.strip()

        symbol_ids = self.get_symbol_ids(symbols)
        num_symbols_of_each_word = self.get_num_symbols_of_each_word(symbols)
        assert len(symbol_ids) == sum(num_symbols_of_each_word)

        # mel, mel_zero_one = self.get_mel(mel_path)
        mel = self.get_mel(mel_path)
        pitch = self.get_pitch(pitch_path)
        energy = self.get_energy(mel)

        input_length = symbol_ids.shape[0]
        mel_length = mel.shape[1]

        attention_prior = get_beta_binomial_prior(input_length, mel_length, scaling_factor=1)

        speaker_id = int(speaker)
        assert speaker_id < self.num_speakers

        return (symbol_ids, num_symbols_of_each_word, mel, pitch, energy, attention_prior, speaker_id)

    def get_num_symbols_of_each_word(self, symbols):
        symbols_of_each_word = symbols.split(" ")
        num_symbols_of_each_word = [len(ws) + 1 for ws in symbols_of_each_word]
        num_symbols_of_each_word[0] = num_symbols_of_each_word[0] + 1
        return num_symbols_of_each_word
    
    def get_symbol_ids(self, symbols):
        symbol_ids = symbols_to_ids(symbols)
        symbol_ids = add_bos_and_eos_ids(symbol_ids)
        symbol_ids = torch.IntTensor(symbol_ids)
        return symbol_ids

    def get_mel(self, mel_path):
        mel = np.load(to_absolute_path(mel_path))
        mel = torch.FloatTensor(mel)
        # mel, mel_zero_one = normalize_mel(mel), normalize_mel(mel, norm_min=0.0, norm_max=1.0)
        # return mel, mel_zero_one
        return mel
    
    def get_pitch(self, pitch_path):
        pitch = np.load(to_absolute_path(pitch_path))
        pitch = torch.FloatTensor(pitch)
        zeros = (pitch == 0.0)
        pitch -= self.pitch_mean
        pitch /= self.pitch_std
        pitch[zeros] = 0.0
        return pitch
    
    def get_energy(self, mel):
        mel = (mel + 4.0) / (2 * 4.0)
        energy = torch.norm(mel.float(), dim=0, p=2)
        return energy

    def __getitem__(self, index):
        return self.get_item(self.metadata[index])

    def __len__(self):
        return len(self.metadata)


class InstantSpeechCollate():

    def __init__(self):
        pass

    def __call__(self, batch):
        
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),dim=0, descending=True)
        max_input_len = input_lengths[0]
        batch_symbol_ids = torch.LongTensor(len(batch), max_input_len)
        batch_symbol_ids.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            symbol_ids = batch[ids_sorted_decreasing[i]][0]
            batch_symbol_ids[i, :symbol_ids.size(0)] = symbol_ids
        
        n_mel_channels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        
        batch_mel = torch.FloatTensor(len(batch), n_mel_channels, max_target_len)
        batch_mel.zero_()
        batch_pitch = torch.FloatTensor(len(batch), 1, max_target_len)
        batch_pitch.zero_()
        batch_energy = torch.FloatTensor(len(batch), 1, max_target_len)
        batch_energy.zero_()
        batch_attention_prior = torch.FloatTensor(len(batch), max_target_len, max_input_len)
        batch_attention_prior.zero_()
        batch_speaker_id = torch.LongTensor(len(batch), 1)
        batch_speaker_id.zero_()
        
        output_lengths = torch.LongTensor(len(batch))
        
        batch_num_symbols_of_each_word = []

        for i in range(len(ids_sorted_decreasing)):
            num_symbols_of_each_word = batch[ids_sorted_decreasing[i]][1]
            mel = batch[ids_sorted_decreasing[i]][2]
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4]
            attention_prior = batch[ids_sorted_decreasing[i]][5]
            speaker_id = batch[ids_sorted_decreasing[i]][6]
            
            batch_mel[i, :, :mel.size(1)] = mel
            batch_pitch[i, 0, :pitch.size(0)] = pitch
            batch_energy[i, 0, :energy.size(0)] = energy
            batch_attention_prior[i, :attention_prior.size(0), :attention_prior.size(1)] = attention_prior
            batch_speaker_id[i, 0] = speaker_id
            output_lengths[i] = mel.size(1)
            batch_num_symbols_of_each_word.append(num_symbols_of_each_word)
        

        return batch_symbol_ids, batch_num_symbols_of_each_word, batch_mel, batch_pitch, batch_energy, \
            batch_attention_prior, batch_speaker_id, input_lengths, output_lengths


class InstantSpeechTextDataset(torch.utils.data.Dataset):
    
    def __init__(self, filelist_path, num_speakers=1):
        self.metadata = self.load_filelist(filelist_path)
        self.num_speakers = num_speakers

    def load_filelist(self, path):
        metadata = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                metadata.append(json.loads(line.strip()))
        return metadata

    def get_item(self, meta):
        symbols = meta["symbols"]
        symbols = symbols.strip()
        symbol_ids = self.get_symbol_ids(symbols)
        num_symbols_of_each_word = self.get_num_symbols_of_each_word(symbols)
        assert len(symbol_ids) == sum(num_symbols_of_each_word)

        speaker_id = random.randint(0, self.num_speakers-1)

        return (symbol_ids, num_symbols_of_each_word, speaker_id)

    def get_num_symbols_of_each_word(self, symbols):
        symbols_of_each_word = symbols.split(" ")
        num_symbols_of_each_word = [len(ws) + 1 for ws in symbols_of_each_word]
        num_symbols_of_each_word[0] = num_symbols_of_each_word[0] + 1
        return num_symbols_of_each_word
    
    def get_symbol_ids(self, symbols):
        symbol_ids = symbols_to_ids(symbols)
        symbol_ids = add_bos_and_eos_ids(symbol_ids)
        symbol_ids = torch.IntTensor(symbol_ids)
        return symbol_ids

    def __getitem__(self, index):
        return self.get_item(self.metadata[index])

    def __len__(self):
        return len(self.metadata)
    

class InstantSpeechTextCollate():
    
    def __init__(self):
        pass

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]

        batch_symbol_ids = torch.LongTensor(len(batch), max_input_len)
        batch_symbol_ids.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            symbol_ids = batch[ids_sorted_decreasing[i]][0]
            batch_symbol_ids[i, :symbol_ids.size(0)] = symbol_ids
        
        batch_speaker_id = torch.LongTensor(len(batch), 1)
        batch_speaker_id.zero_()
        batch_num_symbols_of_each_word = []

        for i in range(len(ids_sorted_decreasing)):
            num_symbols_of_each_word = batch[ids_sorted_decreasing[i]][1]
            speaker_id = batch[ids_sorted_decreasing[i]][2]
            batch_speaker_id[i, 0] = speaker_id
            batch_num_symbols_of_each_word.append(num_symbols_of_each_word)
        
        return batch_symbol_ids, batch_num_symbols_of_each_word, batch_speaker_id, input_lengths