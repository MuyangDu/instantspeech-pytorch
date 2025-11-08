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

_pad = "[pad]"
_unk = "[unk]"
_eos = "[eos]"
_bos = "[bos]"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [_unk, _eos, _bos]


def get_num_symbols():
    return len(symbols)


def symbols_to_ids(symbol_sequence):
    ids = []
    for symbol in symbol_sequence:
        if symbol in symbols:
            ids.append(symbols.index(symbol))
        else:
            ids.append(symbols.index(_unk))
    return ids


def add_bos_and_eos_ids(ids):
    ids = [symbols.index(_bos)] + ids + [symbols.index(_eos)]
    return ids


def add_bos_id(ids):
    ids = [symbols.index(_bos)] + ids
    return ids


def add_eos_id(ids):
    ids = ids + [symbols.index(_eos)]
    return ids


if __name__ == "__main__":
    print(f"num symbols: {get_num_symbols()}")