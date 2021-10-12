import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import librosa
import soundfile as sf
from g2p_en import G2p
from wav2vec2.wrapper import MinimalClassifier

import sys
sys.path.insert(0, 'waveglow/tacotron2')
from layers import TacotronSTFT

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True)
parser.add_argument('--emo_model_dir', type=str, default=None)
parser.add_argument('--filter_length', type=int, default=1024)
parser.add_argument('--hop_length', type=int, default=256)
parser.add_argument('--win_length', type=int, default=1024)
parser.add_argument('--mel_fmin', type=float, default=0)
parser.add_argument('--mel_fmax', type=float, default=8000)
parser.add_argument('--make_test_set', action='store_true')

args = parser.parse_args()

stft = TacotronSTFT(filter_length=args.filter_length,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    sampling_rate=22050,
                    mel_fmin=args.mel_fmin, mel_fmax=args.mel_fmax).cuda()
if args.emo_model_dir:
    emo_model = MinimalClassifier.load_from_checkpoint(args.emo_model_dir,
                                                       strict=False).cuda()
else:
    emo_model = MinimalClassifier().cuda()
emo_model.freeze()
emo_model.eval()
g2p = G2p()

mel_dir = os.path.join(args.outputdir, 'mels')
Path(mel_dir).mkdir(parents=True, exist_ok=True)
emo_reps_dir = os.path.join(args.outputdir, 'emo_reps')
Path(emo_reps_dir).mkdir(parents=True, exist_ok=True)
raw_dir = os.path.join(args.outputdir, '16k_wav')
Path(raw_dir).mkdir(parents=True, exist_ok=True)

metadata = dict()
root = Path(args.datadir)
for text_file in root.rglob('*.txt'):
    encoding = 'utf-16'
    if text_file.stem in ['0011', '0020', '0015']:
        encoding = 'utf-8'
    if text_file.stem in ['0016', '0017']:
        encoding = 'latin-1'
    with open(text_file, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) == 3:
                name, text, emotion = line
                metadata[name] = {
                    'text': text,
                    'emotion': emotion
                }

if not args.make_test_set:
    run_set = [p for p in root.rglob('*.wav') if 'test' not in str(p)]
else:
    run_set = [p for p in root.rglob('*.wav') if 'test' in str(p)]
for audio_name in tqdm(run_set):
    audio, sr = librosa.load(audio_name, sr=None)
    audio_name = audio_name.stem
    if audio_name not in metadata:
        print (audio_name)
        continue
    length = float(len(audio)) / sr
    text = metadata[audio_name]['text']

    melspec = librosa.resample(audio, sr, 22050)
    melspec = np.clip(melspec, -1, 1)
    melspec = torch.cuda.FloatTensor(melspec).unsqueeze(0)
    melspec = stft.mel_spectrogram(melspec).squeeze(0).cpu().numpy()
    _wav = librosa.resample(audio, sr, 16000)
    _wav = np.clip(_wav, -1, 1)
    emo_reps = torch.cuda.FloatTensor(_wav).unsqueeze(0)
    emo_reps = emo_model(emo_reps).squeeze(0).cpu().numpy()

    np.save(os.path.join(mel_dir, audio_name + '.npy'), melspec)
    sf.write(os.path.join(raw_dir, audio_name + '.wav'), _wav, 16000)
    np.save(os.path.join(emo_reps_dir, audio_name + '.npy'), emo_reps)
    phonemes = g2p(text)
    metadata[audio_name].update({
        'length': length,
        'phonemes': phonemes
    })
metadata = {k: v for k, v in metadata.items() if 'length' in v}

with open(os.path.join(args.outputdir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

