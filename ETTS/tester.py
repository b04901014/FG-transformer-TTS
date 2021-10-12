import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ettstransformer import ETTSTransformer
from .baseline import ETTSTransformer_baseline
import sys
import soundfile as sf
from tqdm import tqdm
import os
from g2p_en import G2p
from wav2vec2.wrapper import MinimalClassifier
import random

class ETTSInferenceModel:
    def __init__(self, text_embed_dim, emo_embed_dim, nmels, maxlength,
                 ngst, nlst, model_dim, model_hidden_size, nlayers, nheads,
                 vocoder_ckpt_path, etts_checkpoint, sampledir):
        self._labelset = G2p().phonemes
        self._labelset += [',', '.', ':', ';', '?', '!', '-', '\"', '\'', ' ']
        self.labelset = {x: i for i, x in enumerate(self._labelset)}
        self.g2p = G2p()
        self.sampledir = sampledir
        self.maxlength = maxlength
        self.ngst = ngst
        self.nlst = nlst
        self.model = ETTSTransformer(text_embed_dim, emo_embed_dim, nmels, maxlength, self.labelset,
                                     ngst, nlst,
                                     d_model=model_dim, hidden_size=model_hidden_size,
                                     nlayers=nlayers, nheads=nheads)
        self.emo_model = MinimalClassifier()
        #Extract state dict
        state_dict = torch.load(etts_checkpoint)['state_dict']
        new_state_dict = dict()
        non_parameters = ['tgt_mask', 'pos_txt.p', 'pos_mel.p']
        for k, v in state_dict.items():
            if k.split('.')[0] == 'model':
                k = '.'.join(k.split('.')[1:])
                if k not in non_parameters:
                    new_state_dict[k] = v
        print (self.model.load_state_dict(new_state_dict, strict=False))

        sys.path.insert(0, 'waveglow') #For waveglow vocoder
        waveglow = torch.load(vocoder_ckpt_path)['model']
        self.vocoder = waveglow.remove_weightnorm(waveglow).eval()
        self.model.eval()
        self.model.cuda()
        self.vocoder.cuda()
        self.emo_model.freeze()
        self.emo_model.eval()
        self.emo_model.cuda()

    def synthesize_with_ref(self, text, global_ref_audio, local_ref_audio, outputname, wav_input=False):
        if wav_input:
            g_emo = torch.cuda.FloatTensor(global_ref_audio).unsqueeze(0)
            g_emo = self.emo_model(g_emo)
            l_emo = torch.cuda.FloatTensor(local_ref_audio).unsqueeze(0)
            l_emo = self.emo_model(l_emo)
        else:
            g_emo = torch.cuda.FloatTensor([np.load(global_ref_audio)])
            l_emo = torch.cuda.FloatTensor([np.load(local_ref_audio)])
        _phonemes = self.g2p(text)
        phonemes = [self.labelset['<s>']]
        for i, phoneme in enumerate(_phonemes):
            if phoneme in self.labelset:
                phonemes.append(self.labelset[phoneme])
            else:
                phonemes.append(self.labelset['<unk>'])
        phonemes.append(self.labelset['</s>'])
        phonemes = torch.cuda.LongTensor([phonemes])
        with torch.no_grad():
            mel, attns = self.model.inference(g_emo, l_emo, phonemes, maxlen=self.maxlength, threshold=.5)
            audio = self.vocoder.infer(mel.transpose(1, 2), sigma=0.6) * 32768.0
        audio = audio.squeeze(0).cpu().numpy().astype('int16')
        sf.write(os.path.join(self.sampledir, outputname), audio, 22050)

    def synthesize_with_sample(self, g_emo, text, outputname):
        _phonemes = self.g2p(text)
        phonemes = [self.labelset['<s>']]
        for i, phoneme in enumerate(_phonemes):
            if phoneme in self.labelset:
                phonemes.append(self.labelset[phoneme])
            else:
                phonemes.append(self.labelset['<unk>'])
        phonemes.append(self.labelset['</s>'])
        phonemes = torch.cuda.LongTensor([phonemes])
        with torch.no_grad():
            g_emo = torch.cuda.FloatTensor(g_emo).unsqueeze(0)
            g_emo = self.emo_model(g_emo)
            l_emo = torch.softmax(torch.randn(random.randint(80, 160), self.nlst, device=torch.device('cuda:0')) * 100, -1) * 0.25
            mel, attns = self.model.inference(g_emo, l_emo, phonemes, maxlen=self.maxlength, threshold=.5, ref_wav=False)
            audio = self.vocoder.infer(mel.transpose(1, 2), sigma=0.6) * 32768.0
        audio = audio.squeeze(0).cpu().numpy().astype('int16')
        sf.write(os.path.join(self.sampledir, outputname), audio, 22050)
