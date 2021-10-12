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
parser.add_argument('--pad_silence', type=float, default=0.15)
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
female_training_spkrs = []
male_training_spkrs = []
testing_spkrs = []
with open(os.path.join(args.datadir, 'speaker-info.txt'), 'r') as f:
    for line in f.readlines():
        speaker_info = line.split()
        gender = speaker_info[2]
        spid = speaker_info[0]
        if gender == 'F' and len(female_training_spkrs) < 44:
            female_training_spkrs.append(spid)
        elif gender == 'M' and len(male_training_spkrs) < 44:
            male_training_spkrs.append(spid)
        elif gender != "GENDER":
            testing_spkrs.append(spid)
speakers = testing_spkrs if args.make_test_set else female_training_spkrs + male_training_spkrs

silence_dict = dict()
with open(os.path.join(args.datadir, 'vctk-silences.0.92.txt'), 'r') as f:
    for line in f.readlines():
        out = line.strip().split(' ')
        if len(out) != 3:
            continue
        name, start, end = out
        silence_dict[name] = (float(start), float(end))

for speaker in tqdm(speakers):
    if speaker not in os.listdir(os.path.join(args.datadir, 'txt')):
        #No text avalible
        continue
    audio_names = Path(os.path.join(args.datadir, 'wav48_silence_trimmed', speaker)).rglob('*_mic1.flac')
    audios = []
    for audio_name in audio_names:
        audio, sr = librosa.load(audio_name, sr=None)
        audio_name = audio_name.stem[:-5]

        if audio_name in silence_dict:
            start, end = silence_dict[audio_name]
            audio = audio[int(start * sr): int(end * sr)]
            audio = np.pad(audio, int(args.pad_silence * sr))
        length = float(len(audio)) / sr
        if length < 2.0:
            continue
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
        with open(os.path.join(args.datadir, 'txt', speaker, audio_name + '.txt'), 'r') as f:
            text = f.readline().strip()
        phonemes = g2p(text)
        metadata[audio_name] = {
            'length': length,
            'text': text,
            'phonemes': phonemes
        }

with open(os.path.join(args.outputdir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

