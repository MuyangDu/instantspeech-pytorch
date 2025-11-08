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
import hydra
import torch
import numpy as np
from model_utils import (
    get_config,
    get_model,
    get_vocoder_model
)
from common_utils import (
    update_paths,
    g2p
)
from text_utils import (
    symbols_to_ids,
    add_bos_and_eos_ids,
    add_bos_id,
    add_eos_id
)
from phonemizer.backend import EspeakBackend
import soundfile as sf


def load_model(args):
    update_paths(args)
    tts_args = args
    tts_config = get_config(tts_args)
    acoustic_model, _ = get_model(tts_args, tts_config)
    vocoder_model, vocoder_hparams = get_vocoder_model(tts_args, tts_config)
    acoustic_model = acoustic_model.cuda()
    vocoder_model = vocoder_model.cuda()
    return acoustic_model, vocoder_model, vocoder_hparams


def initialize_vocoder_states():
    states = []
    
    # Input layer state
    states.append(torch.zeros(1, 80, 6).cuda())
    
    # First upsampling block
    states.append(torch.zeros(1, 512, 15).cuda())
    
    # ResBlock states for first upsampling layer
    states.append(torch.zeros(1, 256, 2).cuda())
    states.append(torch.zeros(1, 256, 2).cuda())
    states.append(torch.zeros(1, 256, 6).cuda())
    states.append(torch.zeros(1, 256, 2).cuda())
    states.append(torch.zeros(1, 256, 10).cuda())
    states.append(torch.zeros(1, 256, 2).cuda())
    
    states.append(torch.zeros(1, 256, 6).cuda())
    states.append(torch.zeros(1, 256, 6).cuda())
    states.append(torch.zeros(1, 256, 18).cuda())
    states.append(torch.zeros(1, 256, 6).cuda())
    states.append(torch.zeros(1, 256, 30).cuda())
    states.append(torch.zeros(1, 256, 6).cuda())
    
    states.append(torch.zeros(1, 256, 10).cuda())
    states.append(torch.zeros(1, 256, 10).cuda())
    states.append(torch.zeros(1, 256, 30).cuda())
    states.append(torch.zeros(1, 256, 10).cuda())
    states.append(torch.zeros(1, 256, 50).cuda())
    states.append(torch.zeros(1, 256, 10).cuda())
    
    # Second upsampling block state
    states.append(torch.zeros(1, 256, 15).cuda())
    
    # ResBlock states for second upsampling layer
    states.append(torch.zeros(1, 128, 2).cuda())
    states.append(torch.zeros(1, 128, 2).cuda())
    states.append(torch.zeros(1, 128, 6).cuda())
    states.append(torch.zeros(1, 128, 2).cuda())
    states.append(torch.zeros(1, 128, 10).cuda())
    states.append(torch.zeros(1, 128, 2).cuda())
    
    states.append(torch.zeros(1, 128, 6).cuda())
    states.append(torch.zeros(1, 128, 6).cuda())
    states.append(torch.zeros(1, 128, 18).cuda())
    states.append(torch.zeros(1, 128, 6).cuda())
    states.append(torch.zeros(1, 128, 30).cuda())
    states.append(torch.zeros(1, 128, 6).cuda())
    
    states.append(torch.zeros(1, 128, 10).cuda())
    states.append(torch.zeros(1, 128, 10).cuda())
    states.append(torch.zeros(1, 128, 30).cuda())
    states.append(torch.zeros(1, 128, 10).cuda())
    states.append(torch.zeros(1, 128, 50).cuda())
    states.append(torch.zeros(1, 128, 10).cuda())
    
    # Third upsampling block state
    states.append(torch.zeros(1, 128, 3).cuda())
    
    # ResBlock states for third upsampling layer
    states.append(torch.zeros(1, 64, 2).cuda())
    states.append(torch.zeros(1, 64, 2).cuda())
    states.append(torch.zeros(1, 64, 6).cuda())
    states.append(torch.zeros(1, 64, 2).cuda())
    states.append(torch.zeros(1, 64, 10).cuda())
    states.append(torch.zeros(1, 64, 2).cuda())
    
    states.append(torch.zeros(1, 64, 6).cuda())
    states.append(torch.zeros(1, 64, 6).cuda())
    states.append(torch.zeros(1, 64, 18).cuda())
    states.append(torch.zeros(1, 64, 6).cuda())
    states.append(torch.zeros(1, 64, 30).cuda())
    states.append(torch.zeros(1, 64, 6).cuda())
    
    states.append(torch.zeros(1, 64, 10).cuda())
    states.append(torch.zeros(1, 64, 10).cuda())
    states.append(torch.zeros(1, 64, 30).cuda())
    states.append(torch.zeros(1, 64, 10).cuda())
    states.append(torch.zeros(1, 64, 50).cuda())
    states.append(torch.zeros(1, 64, 10).cuda())
    
    # Fourth upsampling block state
    states.append(torch.zeros(1, 64, 3).cuda())
    
    # ResBlock states for fourth upsampling layer
    states.append(torch.zeros(1, 32, 2).cuda())
    states.append(torch.zeros(1, 32, 2).cuda())
    states.append(torch.zeros(1, 32, 6).cuda())
    states.append(torch.zeros(1, 32, 2).cuda())
    states.append(torch.zeros(1, 32, 10).cuda())
    states.append(torch.zeros(1, 32, 2).cuda())
    
    states.append(torch.zeros(1, 32, 6).cuda())
    states.append(torch.zeros(1, 32, 6).cuda())
    states.append(torch.zeros(1, 32, 18).cuda())
    states.append(torch.zeros(1, 32, 6).cuda())
    states.append(torch.zeros(1, 32, 30).cuda())
    states.append(torch.zeros(1, 32, 6).cuda())
    
    states.append(torch.zeros(1, 32, 10).cuda())
    states.append(torch.zeros(1, 32, 10).cuda())
    states.append(torch.zeros(1, 32, 30).cuda())
    states.append(torch.zeros(1, 32, 10).cuda())
    states.append(torch.zeros(1, 32, 50).cuda())
    states.append(torch.zeros(1, 32, 10).cuda())
    
    # Output layer state
    states.append(torch.zeros(1, 32, 6).cuda())
    
    return states


def mel_to_waveform_with_states(mel, vocoder_model, states):
    with torch.no_grad():
        waveform, states = vocoder_model(mel, states)
    return waveform, states


def run_tts(acoustic_model, vocoder_model, vocoder_hparams, g2p_backend, word, lookbehind_words, lookahead_words, 
            past=None, end_of_sentence=False, vocoder_states=None):
    symbols_of_current_word, symbols_of_lookahead_words = g2p(word, lookbehind_words, lookahead_words, g2p_backend)
    symbols_of_each_word = [symbols_of_current_word] + symbols_of_lookahead_words
    symbols_of_each_word = [symbols + " " for symbols in symbols_of_each_word]

    if end_of_sentence:
        symbols_of_each_word[-1] = symbols_of_each_word[-1].strip()
    num_symbols_of_each_word = [[len(symbols) for symbols in symbols_of_each_word]]
    
    symbols = "".join(symbols_of_each_word)
    symbol_ids = symbols_to_ids(symbols)
    if past is None:
        symbol_ids = add_bos_id(symbol_ids)
        num_symbols_of_each_word[0][0] = num_symbols_of_each_word[0][0] + 1
    if end_of_sentence is True:
        symbol_ids = add_eos_id(symbol_ids)
        num_symbols_of_each_word[0][-1] = num_symbols_of_each_word[0][-1] + 1

    if past is None:
        past = {
            "inputs": torch.zeros((1, 0), device="cuda", dtype=torch.long),
            "num_symbols_of_each_word": [[]],
            "enc_out": None
        }

    inputs = torch.LongTensor(symbol_ids).cuda().unsqueeze(0)

    with torch.inference_mode():
        mel_increment, mel, past = acoustic_model.infer(
            inputs=inputs,
            num_symbols_of_each_word=num_symbols_of_each_word,
            past=past, 
            pace=1.0, 
            max_duration=75, 
            speaker=0,
            final=end_of_sentence
        )

        waveform_chunk, vocoder_states = mel_to_waveform_with_states(mel_increment, vocoder_model, vocoder_states)
        waveform_chunk = waveform_chunk.squeeze(1)
    
    return waveform_chunk, past, vocoder_states


@hydra.main(config_path="configs", config_name="instantspeech_ljspeech", version_base='1.1')
def main(args):
    acoustic_model, vocoder_model, vocoder_hparams = load_model(args)
    
    g2p_backend = EspeakBackend('en-us', preserve_punctuation=True, with_stress=True)
    text = args.text
    words = text.split(" ")
    lookbehind_words = []
    past = None
    
    vocoder_states = initialize_vocoder_states()
    
    index = 0
    waveform_chunks = []
    
    while index < len(words) - 1:
        end_of_sentence = True if index == len(words) - 2 else False
        word = words[index]
        lookahead_words = [words[index + 1]]

        print(f"[{index}] Synthesizing word: {word}. Lookahead words: {' '.join(lookahead_words)}")
        
        waveform_chunk, past, vocoder_states = run_tts(
            acoustic_model=acoustic_model, 
            vocoder_model=vocoder_model, 
            vocoder_hparams=vocoder_hparams,
            g2p_backend=g2p_backend,
            word=word, 
            lookbehind_words=lookbehind_words, 
            lookahead_words=lookahead_words, 
            past=past, 
            end_of_sentence=end_of_sentence,
            vocoder_states=vocoder_states
        )
        
        waveform_chunks.append(waveform_chunk)
        lookbehind_words.append(word)
        index += 1
    
    waveform = torch.cat(waveform_chunks, dim=1).squeeze().cpu().numpy()
    
    fade_in_frames = 5
    fade_in_samples = fade_in_frames * vocoder_hparams.hop_size
    hann_window = np.hanning(fade_in_samples * 2)
    fade_in_coeff = hann_window[:fade_in_samples]
    waveform[:fade_in_samples] *= fade_in_coeff
    
    sf.write("synthesized.wav", waveform, vocoder_hparams.sampling_rate)
    print(f"Synthesized audio saved to synthesized.wav")


if __name__ == "__main__":
    main()