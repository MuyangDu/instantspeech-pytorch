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

import matplotlib.pylab as plt


def plot_alignment_to_figure(alignment, info=None, xlabel='Decoder timestep', ylabel='Encoder timestep'):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig


def plot_spectrogram_to_figure(spectrogram, info=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    xlabel = "Frames"
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig


def plot_to_figure(target, predicted, info):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(target)), target, alpha=1,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(predicted)), predicted, alpha=1,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel(f"{info}")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig