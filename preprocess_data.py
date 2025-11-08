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

import argparse
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from tqdm import tqdm

import json
import numpy as np
import librosa
import random
import torch
import os
from phonemizer.backend import EspeakBackend
from types import SimpleNamespace
from config import Config
from audio_utils import AudioUtils
import string


def estimate_pitch(signal, hop_size, method="pyin"):
    if method == "pyin":
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), 
            frame_length=hop_size * 4
        )
    else:
        f0 = librosa.yin(
            signal, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), 
            frame_length=hop_size * 4
        )
    f0 = np.where(np.isnan(f0), 0.0, f0)
    return f0


def write_lines_to_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def extract_mel_and_pitch(args):
    audio_path, mel_path, pitch_path, vocoder_hparams, audio_utils = args
    success = False
    try:
        wav, sr = librosa.load(audio_path, sr=vocoder_hparams.sampling_rate, mono=True)
        pitch = estimate_pitch(wav, hop_size=vocoder_hparams.hop_size, method="yin")

        if vocoder_hparams.name == "causal_hifigan":
            mel = audio_utils.mel_spectrogram(wav)
        else:
            raise NotImplementedError

        pitch = pitch[:mel.shape[1]]
        assert pitch.shape[0] == mel.shape[1]
        np.save(mel_path, mel)
        np.save(pitch_path, pitch)
        success = True
    except Exception as e:
        print(repr(e))
    return mel_path, pitch_path, audio_path, success


def process_hifitts(dataset_dir, vocoder_hparams, audio_utils):
    mp.set_start_method("spawn", force=True)

    mel_dir = "data/hifitts/mel"
    pitch_dir = "data/hifitts/pitch"
    train_filelist = "data/hifitts/train_filelist.txt"
    dev_filelist = "data/hifitts/dev_filelist.txt"
    test_filelist = "data/hifitts/test_filelist.txt"
    audio_dir = os.path.join(dataset_dir, "audio")

    manifest_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    speakers = []
    for i in os.listdir(audio_dir):
        j = os.listdir(os.path.join(audio_dir, i))
        for k in j:
            speakers.append(f"{i}/{k}")
    print(f"Num Speakers: {len(speakers)}")

    def load_manifest(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()]

    backend = EspeakBackend('en-us', preserve_punctuation=True, with_stress=True)
    output_items = []

    for f in tqdm(manifest_files):
        manifest_path = os.path.join(dataset_dir, f)
        items = load_manifest(manifest_path)
        if "train" in f:
            data_type = "train"
        elif "dev" in f:
            data_type = "dev"
        else:
            data_type = "test"

        for item in items:
            text_normalized = item["text_normalized"]
            audio_path = os.path.join(dataset_dir, item["audio_filepath"])
            audio_name = os.path.basename(audio_path)

            symbols = backend.phonemize([text_normalized], strip=True)[0]
            speaker = item["audio_filepath"].split("/")[1] + "/" + item["audio_filepath"].split("/")[2]
            speaker_id = speakers.index(speaker)
            mel_path = os.path.join(mel_dir, audio_name.replace("flac", "npy"))
            pitch_path = os.path.join(pitch_dir, audio_name.replace("flac", "npy"))

            json_item = {
                "text": text_normalized,
                "symbols": symbols,
                "audio": audio_path,
                "mel": mel_path,
                "pitch": pitch_path,
                "speaker": speaker_id,
                "type": data_type
            }
            output_items.append(json_item)

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)

    args_list = [(item["audio"], item["mel"], item["pitch"], vocoder_hparams, audio_utils) for item in output_items]

    with Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(extract_mel_and_pitch, args_list), total=len(args_list)))

    pitch = [np.load(item[1]) for item in results if item[3]]
    pitch = np.concatenate(pitch) if pitch else np.array([])
    if pitch.size > 0:
        np.save(os.path.join(pitch_dir, "pitch_mean.npy"), np.mean(pitch))
        np.save(os.path.join(pitch_dir, "pitch_std.npy"), np.std(pitch))

    train_output_lines = []
    dev_output_lines = []
    test_output_lines = []

    for item in output_items:
        if os.path.exists(item["mel"]) and os.path.exists(item["pitch"]):
            if item["type"] == "train":
                train_output_lines.append(json.dumps(item))
            elif item["type"] == "dev":
                dev_output_lines.append(json.dumps(item))
            else:
                test_output_lines.append(json.dumps(item))

    random.shuffle(train_output_lines)
    random.shuffle(dev_output_lines)
    random.shuffle(test_output_lines)
    write_lines_to_file(train_filelist, train_output_lines)
    write_lines_to_file(dev_filelist, dev_output_lines)
    write_lines_to_file(test_filelist, test_output_lines)


def process_ljspeech(dataset_dir, vocoder_hparams, audio_utils):
    mp.set_start_method("spawn", force=True)
    
    mel_dir = "data/ljspeech/mel"
    pitch_dir = "data/ljspeech/pitch"
    train_filelist = "data/ljspeech/train_filelist.txt"
    dev_filelist = "data/ljspeech/dev_filelist.txt"
    test_filelist = "data/ljspeech/test_filelist.txt"
    audio_dir = os.path.join(dataset_dir, "wavs")

    metadata_file = os.path.join(dataset_dir, "metadata.csv")
    
    def load_metadata(path):
        with open(path, encoding="utf-8") as f:
            return f.readlines()

    metadata = load_metadata(metadata_file)
    
    backend = EspeakBackend('en-us', preserve_punctuation=True, with_stress=True)
    output_items = []
    
    for line in metadata:
        audio_name, text, text_normalized = line.strip().split("|")
        audio_path = os.path.join(dataset_dir, "wavs", f"{audio_name}.wav")
        symbols = backend.phonemize([text_normalized], strip=True)[0]
        speaker_id = 0
        mel_path = os.path.join(mel_dir, f"{audio_name}.npy")
        pitch_path = os.path.join(pitch_dir, f"{audio_name}.npy")

        json_item = {
            "text": text_normalized,
            "symbols": symbols,
            "audio": audio_path,
            "mel": mel_path,
            "pitch": pitch_path,
            "speaker": speaker_id
        }
        output_items.append(json_item)

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    
    args_list = [(item["audio"], item["mel"], item["pitch"], vocoder_hparams, audio_utils) for item in output_items]
    
    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(extract_mel_and_pitch, args_list), total=len(args_list)))
    
    pitch = [np.load(item[1]) for item in results if item[3]]
    pitch = np.concatenate(pitch) if pitch else np.array([])
    if pitch.size > 0:
        np.save(os.path.join(pitch_dir, "pitch_mean.npy"), np.mean(pitch))
        np.save(os.path.join(pitch_dir, "pitch_std.npy"), np.std(pitch))
    
    output_lines = []
    train_output_lines = []
    dev_output_lines = []
    test_output_lines = []

    for item in output_items:
        if os.path.exists(item["mel"]) and os.path.exists(item["pitch"]):
            output_lines.append(json.dumps(item))
    
    random.shuffle(output_lines)
    train_output_lines = output_lines[:-200]
    dev_output_lines = output_lines[-200:-100]
    test_output_lines = output_lines[-100:]
    write_lines_to_file(train_filelist, train_output_lines)
    write_lines_to_file(dev_filelist, dev_output_lines)
    write_lines_to_file(test_filelist, test_output_lines)


def process_ultrachat(dataset_dir):
    import nltk
    nltk.download('punkt_tab')

    min_length = 10
    max_length = 200

    jsonl_file = os.path.join(dataset_dir, "train_0.jsonl")
    train_filelist = f"data/ultrachat/train_filelist_{min_length}_{max_length}.txt"
    dev_filelist = f"data/ultrachat/dev_filelist_{min_length}_{max_length}.txt"
    test_filelist = f"data/ultrachat/test_filelist_{min_length}_{max_length}.txt"

    backend = EspeakBackend('en-us', preserve_punctuation=True, with_stress=True)

    def load_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()]
    
    def get_sentences(items):
        sentences = []
        for item in tqdm(items):
            sentences.extend(nltk.sent_tokenize(" ".join(item["data"]).replace("\n", " ")))

        return sentences

    def is_english_text(s):
        allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)
        return all(char in allowed_chars for char in s)
        
    items = load_jsonl(jsonl_file)
    sentences = get_sentences(items)
    sentences = [s for s in sentences if len(s) >= min_length and len(s) <= max_length and is_english_text(s)]

    output_items = []
    for sentence in tqdm(sentences):
        symbols = backend.phonemize([sentence], strip=True)[0]
        json_item = {
            "text": sentence,
            "symbols": symbols
        }
        output_items.append(json_item)

    output_lines = [json.dumps(item) for item in output_items]
    random.shuffle(output_lines)
    train_output_lines = output_lines[:-2000]
    dev_output_lines = output_lines[-2000: -1000]
    test_output_lines = output_lines[-1000:]

    os.makedirs(os.path.dirname(train_filelist), exist_ok=True)
    os.makedirs(os.path.dirname(dev_filelist), exist_ok=True)
    os.makedirs(os.path.dirname(test_filelist), exist_ok=True)

    write_lines_to_file(train_filelist, train_output_lines)
    write_lines_to_file(dev_filelist, dev_output_lines)
    write_lines_to_file(test_filelist, test_output_lines)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', choices=["ljspeech", "hifitts", "ultrachat"], required=True)
    parser.add_argument('--dataset-dir', required=True, type=str)
    parser.add_argument('--vocoder-model', type=str, default="causal_hifigan")
    parser.add_argument('--config', type=str, default="configs/config.json")
    
    args = parser.parse_args()

    if args.vocoder_model.startswith("causal_hifigan"):
        config = Config.from_json(args.config)
        vocoder_hparams = SimpleNamespace(name="causal_hifigan", sampling_rate=config.sample_rate, hop_size=config.stft_hop_length)
        audio_utils = AudioUtils(config)
    else:
        raise NotImplementedError
    
    if args.dataset_name == "ljspeech":
        process_ljspeech(args.dataset_dir, vocoder_hparams, audio_utils)
    elif args.dataset_name == "hifitts":
        process_hifitts(args.dataset_dir, vocoder_hparams, audio_utils)
    elif args.dataset_name == "ultrachat":
        process_ultrachat(args.dataset_dir)