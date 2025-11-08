# InstantSpeech

<p>
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/pytorch-2.5.1-ee4c2c?logo=pytorch" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/license-Apache-green" alt="License">
  <img src="https://img.shields.io/badge/model-InstantSpeech-orange" alt="Model">
  <a href="https://ieeexplore.ieee.org/abstract/document/10890120">
    <img src="https://img.shields.io/badge/IEEE-Paper-blue?logo=ieee" alt="IEEE Paper">
  </a>
</p>


PyTorch implementation of the paper: **“InstantSpeech: Instant Synchronous Text-to-Speech Synthesis for LLM-driven Voice Chatbots”**

InstantSpeech is a lightweight, fully parallel, low-latency text-to-speech synthesis model designed for real-time voice chatbots powered by large language models (LLMs). Unlike traditional TTS models that wait for complete sentences, InstantSpeech enables instant word-by-word streaming speech synthesis, starting immediately after the LLM generates just a few words.

With InstantSpeech, you can easily enable any text-based LLM to produce streaming audio outputs directly, without waiting for the text generation to complete.

---

## Environment Setup

We use conda for environment management.

```bash
conda create -n instantspeech python=3.10.16
conda activate instantspeech
apt update && apt install libsndfile1-dev espeak-ng -y
pip install -r requirements.txt
```

---

## Dataset Preparation

### 1. Speech Dataset (for Pretraining)

We use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) as an example dataset for model pretraining. Please manually download the LJSpeech dataset and place it under:

```
data/ljspeech/
```

After extraction, the dataset directory should look like:

```
./data/ljspeech/LJSpeech-1.1
```

Then, run the preprocessing script to extract mel spectrograms, pitch, and generate filelists.  

```bash
python preprocess_data.py   --dataset-name=ljspeech   --dataset-dir=./data/ljspeech/LJSpeech-1.1
```

---

### 2. Text-Only Dataset (for Distillation)

For the distillation stage, we use [UltraChat](https://huggingface.co/datasets/stingning/ultrachat/tree/main) as an example of a text-only dataset. Download `train_0.jsonl` from the UltraChat dataset and place it under:

```
data/ultrachat/
```

Then run:

```bash
python preprocess_data.py   --dataset-name=ultrachat   --dataset-dir=./data/ultrachat
```

---

## Vocoder Preparation

A pre-trained Causal HiFi-GAN vocoder checkpoint is required for model evaluation during InstantSpeech training. The training recipe for our Causal HiFi-GAN will be released soon. For now, please fill out this [form](https://forms.cloud.microsoft/r/WAZkyAXmW6) to obtain a pre-trained Causal HiFi-GAN checkpoint.

Once you received the checkpoint, place it under:

```
cktps/vocoder.pt
```

The pre-trained checkpoint is **universal** so you can directly use it to train InstantSpeech on your own datasets.

---

## Model Training

### Train the Lookahead-Constrained (Student) Model

```bash
bash train.sh
```

### Train the Unconstrained (Teacher) Model

```bash
bash train_teacher.sh
```

### Optional: Distillation from Teacher to Student

To improve the naturalness of the student model, you can distill it using the teacher model with the text-only dataset:

```bash
bash distill.sh
```

If you find that the speech synthesized by the pretrained model is already sufficiently natural, you may skip the distillation step.

---

## Monitoring Training Logs

You can monitor training and distillation progress using TensorBoard:

```bash
tensorboard --logdir=./logs --port=6007
```

In TensorBoard, you can visualize loss curves and listen to generated audio samples.

---

## Inference

Run the inference script to synthesize speech word-by-word:

```bash
bash inference.sh
```

---

## Checkpoints

To obtain pre-trained checkpoints for testing InstantSpeech, please fill out this [form](https://forms.cloud.microsoft/r/WAZkyAXmW6).

---

## Citation

If you use InstantSpeech in your research or project, please cite the following paper:

```bibtex
@inproceedings{du2025instantspeech,
  title={InstantSpeech: Instant Synchronous Text-to-Speech Synthesis for LLM-driven Voice Chatbots},
  author={Du, Muyang and Liu, Chuan and Lai, Junjie},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

---

## Contact

If you have any questions, issues, or need technical support, please feel free to contact us by filling out this [form](https://forms.cloud.microsoft/r/WAZkyAXmW6).

---

## Acknowledgements

- The training framework is inspired by [nanoT5](https://github.com/PiotrNawrot/nanoT5).  
- The Causal HiFi-GAN vocoder is modified from [HiFi-GAN](https://github.com/jik876/hifi-gan).