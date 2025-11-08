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

import librosa
import scipy
import numpy as np
import soundfile as sf


class AudioUtils:

    def __init__(self, config):
        self.config = config
        self.mel_basis = None
        self.inverse_mel_basis = None


    def load_wav(self, path, sr):
        signal, sr = librosa.load(path, sr=sr)
        return signal, sr
        
        
    def save_wav(self, path, signal, sr):
        sf.write(path, signal, sr, "PCM_16")


    def resample(self, signal, original_sr, target_sr):
        resampled_sr = librosa.resample(y=signal, orig_sr=original_sr, target_sr=target_sr)
        return resampled_sr, target_sr

    
    def pre_emphasis(self, signal):
        signal = scipy.signal.lfilter([1, -self.config.pre_emphasis], [1], signal)
        return signal


    def inverse_pre_emphasis(self, signal):
        signal = scipy.signal.lfilter([1], [1, -self.config.pre_emphasis], signal)
        return signal


    def stft(self, signal):
        complex_valued_matrix = librosa.stft(
            y=signal,
            n_fft=self.config.num_fft_components,
            win_length=self.config.stft_window_length,
            hop_length=self.config.stft_hop_length
        )
        return complex_valued_matrix

    
    def inverse_stft(self, stft_matrix):
        signal = librosa.istft(
            stft_matrix, 
            hop_length=self.config.stft_hop_length, 
            win_length=self.config.stft_window_length
        )
        return signal
        
        
    def spectrogram(self, signal):
        if self.config.pre_emphasis != 0:
            signal = self.pre_emphasis(signal)
        complex_valued_matrix = self.stft(signal)
        spectrogram = self.amplitude_to_decibel(
            np.abs(complex_valued_matrix)) - self.config.reference_level_decibel
        if self.config.spectrogram_normalization:
            normalized_spectrogram = self.normalize(spectrogram)
            return normalized_spectrogram
        else:
            return spectrogram

    
    def mel_spectrogram(self, signal):
        if self.config.pre_emphasis != 0:
            signal = self.pre_emphasis(signal)
        complex_valued_matrix = self.stft(signal)
        mel_spectrogram = self.amplitude_to_decibel(
            self.linear_to_mel(np.abs(complex_valued_matrix))) - self.config.reference_level_decibel
        if self.config.spectrogram_normalization:
            normalized_mel_spectrogram = self.normalize(mel_spectrogram)
            return normalized_mel_spectrogram
        else:
            return mel_spectrogram

    
    def mel_spectrogram_zero_one_norm(self, signal):
        if self.config.pre_emphasis != 0:
            signal = self.pre_emphasis(signal)
        complex_valued_matrix = self.stft(signal)
        mel_spectrogram = self.amplitude_to_decibel(
            self.linear_to_mel(np.abs(complex_valued_matrix))) - self.config.reference_level_decibel
        normalized_mel_spectrogram = self.zero_one_normalize(mel_spectrogram)
        return normalized_mel_spectrogram


    def linear_to_mel(self, spectrogram):
        if self.mel_basis is None:
            self.mel_basis = self.get_mel_basis()
        return np.dot(self.mel_basis, spectrogram)

    
    def mel_to_linear(self, mel_spectrogram):
        if self.inverse_mel_basis is None:
            self.inverse_mel_basis = self.get_inverse_mel_basis()
        return np.maximum(1e-10, np.dot(self.inverse_mel_basis, mel_spectrogram))


    def get_mel_basis(self):
        # highest frequency should be smaller than sr // 2
        assert self.config.highest_frequency <= self.config.sample_rate // 2
        mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.num_fft_components,
            n_mels=self.config.n_mel_channels,
            fmin=self.config.lowest_frequency,
            fmax=self.config.highest_frequency
        )
        return mel_basis


    def get_inverse_mel_basis(self):
        if self.mel_basis is None:
            self.mel_basis = self.get_mel_basis()
        return np.linalg.pinv(self.mel_basis)


    def amplitude_to_decibel(self, amplitude):
        minimum_level_amplitude = self.decibel_to_amplitude(self.config.minimum_level_decibel)
        decibel = 20 * np.log10(np.maximum(minimum_level_amplitude, amplitude))
        return decibel


    def decibel_to_amplitude(self, decibel):
        amplitude = np.power(10.0, decibel * 0.05)
        return amplitude


    def zero_one_normalize(self, spectrogram):
        normalized_spectrogram = np.clip((spectrogram - self.config.minimum_level_decibel) \
            / -self.config.minimum_level_decibel, 0, 1)
        return normalized_spectrogram


    def normalize(self, spectrogram):
        if self.config.symmetric_mels:
            normalized_spectrogram = (spectrogram - self.config.minimum_level_decibel) / -self.config.minimum_level_decibel
            normalized_spectrogram = (2 * self.config.max_abs_value) * normalized_spectrogram - self.config.max_abs_value
            if self.config.allow_clip_in_normalization:
                normalized_spectrogram = np.clip(normalized_spectrogram, -self.config.max_abs_value, self.config.max_abs_value)
        else:
            normalized_spectrogram = (spectrogram - self.config.minimum_level_decibel) / -self.config.minimum_level_decibel
            normalized_spectrogram = self.config.max_abs_value * normalized_spectrogram
            if self.config.allow_clip_in_normalization:
                normalized_spectrogram = np.clip(normalized_spectrogram, 0, self.config.max_abs_value)
        return normalized_spectrogram

    
    def denormalize(self, normalized_spectrogram):
        if self.config.symmetric_mels:
            if self.config.allow_clip_in_normalization:
                spectrogram = np.clip(normalized_spectrogram, -self.config.max_abs_value, self.config.max_abs_value)
            spectrogram = (spectrogram + self.config.max_abs_value) / (2 * self.config.max_abs_value)
            spectrogram = spectrogram * (-self.config.minimum_level_decibel) + self.config.minimum_level_decibel
        else:
            if self.config.allow_clip_in_normalization:
                spectrogram = np.clip(normalized_spectrogram, 0, self.config.max_abs_value)
            spectrogram = spectrogram / self.config.max_abs_value
            spectrogram = spectrogram * (-self.config.minimum_level_decibel) + self.config.minimum_level_decibel
        return spectrogram
    

    def griffin_lim(self, spectrogram):
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
        S_complex = np.abs(spectrogram).astype(np.complex)
        signal = self.inverse_stft(S_complex * angles)
        for i in range(self.config.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self.stft(signal)))
            signal = self.inverse_stft(S_complex * angles)
        return signal

    
    def mel_to_signal(self, mel_spectrogram):
        if self.config.spectrogram_normalization:
            denormalized_mel_spectrogram = self.denormalize(mel_spectrogram)
        else:
            denormalized_mel_spectrogram = mel_spectrogram
        spectrogram = self.mel_to_linear(
            self.decibel_to_amplitude(denormalized_mel_spectrogram + \
                self.config.reference_level_decibel)
        )
        emphasized_signal = self.griffin_lim(spectrogram ** self.config.power)
        signal = self.inverse_pre_emphasis(emphasized_signal)
        return signal

    
    def linear_to_signal(self, spectrogram):
        if self.config.spectrogram_normalization:
            denormalized_spectrogram = self.denormalize(spectrogram)
        else:
            denormalized_spectrogram = spectrogram
        emphasized_signal = self.griffin_lim(denormalized_spectrogram ** self.config.power)
        signal = self.inverse_pre_emphasis(emphasized_signal)
        return signal