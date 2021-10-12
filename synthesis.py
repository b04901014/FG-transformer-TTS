from ETTS.tester import ETTSInferenceModel
import argparse
import shutil
import os
import json
import numpy as np
import random
import librosa
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--maxlength', type=int, default=1000)
parser.add_argument('--nmels', type=int, default=80)
parser.add_argument('--emo_embed_dim', type=int, default=768)
parser.add_argument('--text_embed_dim', type=int, default=256)
parser.add_argument('--model_dim', type=int, default=256)
parser.add_argument('--model_hidden_size', type=int, default=512*2)
parser.add_argument('--nlayers', type=int, default=5)
parser.add_argument('--nheads', type=int, default=2)
parser.add_argument('--ngst', type=int, default=64)
parser.add_argument('--nlst', type=int, default=32)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--etts_checkpoint', type=str, required=True)
parser.add_argument('--datatype', type=str, required=True, choices=['LJSpeech', 'VCTK'])
parser.add_argument('--vocoder_ckpt_path', type=str, required=True)
parser.add_argument('--sampledir', type=str, required=True)
args = parser.parse_args()
model = ETTSInferenceModel(text_embed_dim=args.text_embed_dim,
                           emo_embed_dim=args.emo_embed_dim,
                           nmels=args.nmels,
                           maxlength=args.maxlength,
                           ngst=args.ngst,
                           nlst=args.nlst,
                           model_dim=args.model_dim,
                           model_hidden_size=args.model_hidden_size,
                           nlayers=args.nlayers,
                           nheads=args.nheads,
                           vocoder_ckpt_path=args.vocoder_ckpt_path,
                           etts_checkpoint=args.etts_checkpoint,
                           sampledir=args.sampledir)


text = 'This is a very difficult out of domain utterance ready to be synthesized by our system.'

#LJSpeech
if args.datatype == 'LJSpeech':
    #Use whatever speaker reference speech you want, or simply use the average of all samples in the training set
    #It doesn't matter since it is single speaker
    global_audio = 'Dataset/LJSpeech-1.1/processed/16k_wav/LJ001-0024.wav'
    global_audio, sr = librosa.load(global_audio, sr=None) #16000
    assert sr == 16000

    #Condition on style reference audio
    local_audio = 'Dataset/LJSpeech-1.1/processed/16k_wav/LJ001-0025.wav'
    local_audio, sr = librosa.load(local_audio, sr=None)
    assert sr == 16000

    model.synthesize_with_ref(text, global_audio, local_audio, f'output_1.wav', True)

    #Random sample LST
    model.synthesize_with_sample(global_audio, text, f'output_2.wav')

#VCTK
else:
    #Condition on both speaker and style reference speech
    global_audio = 'Dataset/VCTK/processed_test/16k_wav/p335_018.wav'
    global_audio, sr = librosa.load(global_audio, sr=None) #16000
    assert sr == 16000
    local_audio = 'Dataset/VCTK/processed_test/16k_wav/p317_030.wav'
    local_audio, sr = librosa.load(local_audio, sr=None)
    assert sr == 16000
    model.synthesize_with_ref(text, global_audio, local_audio, f'output_1.wav', True)

    #Random sample style reference speech
    model.synthesize_with_sample(global_audio, text, f'output_2.wav')
