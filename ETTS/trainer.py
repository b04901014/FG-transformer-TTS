import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .dataloader import LJSpeechDataset, VCTKDataset, RandomBucketSampler
from torch.utils import data
import pytorch_lightning.core.lightning as pl
import sys
import soundfile as sf
from tqdm import tqdm
import os
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class ETTSTrasnformerModel(pl.LightningModule):
    def __init__(self, datadir, datatype, vocoder_ckpt_path, maxlength, nmels,
                 emo_embed_dim, text_embed_dim, model_dim, model_hidden_size,
                 nlayers, nheads, ngst, nlst, use_lst,
                 train_bucket_size, val_bucket_size,
                 warmup_step, maxstep, lr, batch_size,
                 distributed, val_split, gate_pos_weight,
                 nworkers, num_audio_sample, sampledir, n_attn_plots,
                 use_guided_attn, n_guided_steps,
                 etts_checkpoint=None):
        super().__init__()
        self.datatype = datatype
        self.nworkers = nworkers
        self.train_bucket_size = train_bucket_size
        self.val_bucket_size = val_bucket_size
        self.warmup_step = warmup_step
        self.maxstep = maxstep
        self.lr = lr
        self.maxlength = maxlength
        self.distributed  = distributed
        self.num_audio_sample = num_audio_sample
        self.sampledir = sampledir
        self.datadir = datadir
        self.batch_size = batch_size
        self.n_attn_plots = n_attn_plots
        self.use_guided_attn = use_guided_attn
        self.n_guided_steps = n_guided_steps
        self.use_lst = use_lst
        self.data = LJSpeechDataset(datadir) if self.datatype == 'LJSpeech' else VCTKDataset(datadir)
        if self.use_lst:
            from .ettstransformer import ETTSTransformer
            self.model = ETTSTransformer(text_embed_dim, emo_embed_dim, nmels, maxlength, self.data.labelset,
                                         ngst, nlst,
                                         d_model=model_dim, hidden_size=model_hidden_size,
                                         nlayers=nlayers, nheads=nheads)
        else:
            from .baseline import ETTSTransformer_baseline
            self.model = ETTSTransformer_baseline(text_embed_dim, emo_embed_dim, nmels, maxlength, self.data.labelset, ngst,
                                                  d_model=model_dim, hidden_size=model_hidden_size,
                                                  nlayers=nlayers, nheads=nheads)

        if etts_checkpoint is not None:
            #Extract state dict
            state_dict = torch.load(etts_checkpoint)['state_dict']
            non_parameters = ['tgt_mask', 'pos_txt.p', 'pos_mel.p']
            new_state_dict = dict()
            for k, v in state_dict.items():
                if k.split('.')[0] == 'model':
                    k = '.'.join(k.split('.')[1:])
                    if k not in non_parameters:
                        new_state_dict[k] = v
            print (self.model.load_state_dict(new_state_dict, strict=False))

        sys.path.insert(0, 'waveglow') #For waveglow vocoder
        waveglow = torch.load(vocoder_ckpt_path)['model']
        self.vocoder = waveglow.remove_weightnorm(waveglow).eval()

        numtraining = int(len(self.data) * val_split)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))
        self.register_buffer('gate_pos_weight', torch.FloatTensor([gate_pos_weight]))
        self.gate_criterion = nn.BCEWithLogitsLoss(pos_weight=self.gate_pos_weight)
        self.reconstruct_criterion = nn.L1Loss()

    def train_dataloader(self):
        idxs = self.traindata.indices
        length = [self.data.lengths[i] for i in idxs]
        sampler = RandomBucketSampler(self.train_bucket_size, length, self.batch_size, drop_last=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank, dynamic_batch=False)
        return data.DataLoader(self.traindata,
                               num_workers=self.nworkers,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def val_dataloader(self):
        idxs = self.valdata.indices
        length = [self.data.lengths[i] for i in idxs]
        with open('valid_LJ.set', 'w') as f:
            for i in idxs:
                f.write(f"{self.data.audio_names[i]}\n")
        print ([self.data.audio_names[i] for i in idxs])
        sampler = RandomBucketSampler(self.val_bucket_size, length, self.batch_size, drop_last=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank, dynamic_batch=False)
        return data.DataLoader(self.valdata,
                               num_workers=self.nworkers,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = optim.Adam(params, lr=self.lr)
        #Learning rate scheduler
        num_training_steps = self.maxstep
        num_warmup_steps = self.warmup_step
        num_flat_steps = int(0.3 * num_training_steps)
        def lambda_lr(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
            )
        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lambda_lr),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.datatype == 'LJSpeech':
            mels, l_mel, emos, l_emo, phonemes, l_phoneme, _ = batch
            g_emos, g_l_emos = emos, l_emo
        elif self.datatype == 'VCTK':
            mels, l_mel, emos, l_emo, g_emos, g_l_emos, phonemes, l_phoneme, _ = batch
        if self.use_lst:
            (mel_out, mel_out_post, gate_logit, mask, gloss, _, _, _) = self.model(g_emos, emos, phonemes, mels, g_l_emos, l_emo, l_phoneme, l_mel)
        else:
            (mel_out, mel_out_post, gate_logit, mask, gloss, _, _, _) = self.model(g_emos, phonemes, mels, g_l_emos, l_phoneme, l_mel)
        mask = ~mask
        reconstruct_loss = (self.reconstruct_criterion(mel_out[mask], mels[mask]) +
                            self.reconstruct_criterion(mel_out_post[mask], mels[mask]))
        gate_labels = F.one_hot(l_mel - 1, num_classes=mels.size(1)).float()
        gate_loss = self.gate_criterion(gate_logit[mask], gate_labels[mask])
        loss = reconstruct_loss + gate_loss
        if self.use_guided_attn and self.global_step < self.n_guided_steps:
            loss += gloss

        tqdm_dict = {
            'loss': loss,
            'rec_loss': reconstruct_loss,
            'gate_loss': gate_loss,
            'guided_loss': gloss
        }
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.sampled_emotion, self.sampled_text = [], []
        self.reference_names, self.text_names = [], []
        self.sampled_attention_plots = 0

    def validation_step(self, batch, batch_idx):
        if self.datatype == 'LJSpeech':
            mels, l_mel, emos, l_emo, phonemes, l_phoneme, names = batch
            g_emos, g_l_emos = emos, l_emo
        elif self.datatype == 'VCTK':
            mels, l_mel, emos, l_emo, g_emos, g_l_emos, phonemes, l_phoneme, names = batch
        if self.use_lst:
            mel_out, mel_out_post, gate_logit, mask, gloss, attns, enc_attn, dec_attn = self.model(g_emos, emos, phonemes, mels, g_l_emos, l_emo, l_phoneme, l_mel)
        else:
            (mel_out, mel_out_post, gate_logit, mask, gloss, attns, enc_attn, dec_attn) = self.model(g_emos, phonemes, mels, g_l_emos, l_phoneme, l_mel)
        mask = ~mask
        reconstruct_loss = (self.reconstruct_criterion(mel_out[mask], mels[mask]) +
                            self.reconstruct_criterion(mel_out_post[mask], mels[mask]))
        gate_labels = F.one_hot(l_mel - 1, num_classes=mels.size(1)).float()
        gate_loss = self.gate_criterion(gate_logit[mask], gate_labels[mask])
        loss = reconstruct_loss + gate_loss
        if self.use_guided_attn and self.global_step < self.n_guided_steps:
            loss += gloss

        validdict = {
            'val_loss': loss,
            'val_rec_loss': reconstruct_loss,
            'val_gate_loss': gate_loss,
            'val_guided_loss': gloss
        }
        self.log_dict(validdict, on_epoch=True, logger=True, sync_dist=self.distributed)
        if len(self.sampled_text) < self.num_audio_sample and batch_idx % 2 == 0:
            self.sampled_text += [text[: l] for text, l in zip(phonemes, l_phoneme)]
            self.text_names += names
        if len(self.sampled_emotion) < self.num_audio_sample and batch_idx % 2 == 1:
            self.sampled_emotion += [emo[: l] for emo, l in zip(emos.detach(), l_emo)]
            self.reference_names += names
        if self.sampled_attention_plots < self.n_attn_plots:
            self.plot_attn(attns, 'ed')
            self.plot_attn(enc_attn, 'enc')
            self.plot_attn(dec_attn, 'dec')
            self.sampled_attention_plots += 1
        return loss

    def plot_attn(self, attns, prefix):
        fig, axs = plt.subplots(1, len(attns))
        for i, attn in enumerate(attns): #Each layers
            attn = attn.cpu().numpy()
            sampled_attn = attn[0]
            axs[i].matshow(sampled_attn)
        outpath = os.path.join(self.sampledir, f'epoch{self.current_epoch}-{prefix}-{self.sampled_attention_plots}.png')
        fig.savefig(outpath)
        fig.clf()
        plt.close()

    def on_validation_epoch_end(self):
        if self.trainer.local_rank != 0:
            return
        print ("Synthesizing Audio with unpaired emo/text from sampled validation set...")
        #Run with mis-paired reference audio / text
        self.sampled_attention_plots = 0
        #Use the first text as text condition, observe the difference when inputting different reference speech
        text = self.sampled_text[0].unsqueeze(0)
        ref_text_name = self.text_names[0]
        ref_text = self.data.label[ref_text_name]['text']
        with open(os.path.join(self.sampledir, f'epoch{self.current_epoch}.txt'), 'w') as f:
            f.write(ref_text)
        for i in tqdm(range(min(self.num_audio_sample, len(self.sampled_text)))):
            emo = self.sampled_emotion[i].unsqueeze(0)
            with torch.no_grad():
                if self.use_lst:
                    mel, attns = self.model.inference(emo, emo, text, maxlen=self.maxlength, threshold=.5)
                else:
                    mel, attns = self.model.inference(emo, text, maxlen=self.maxlength, threshold=.5)
                audio = self.vocoder.infer(mel.transpose(1, 2), sigma=0.6) * 32768.0
            audio = audio.squeeze(0).cpu().numpy().astype('int16')
            sf.write(os.path.join(self.sampledir, f'epoch{self.current_epoch}-{i}.wav'), audio, 22050)

            if self.sampled_attention_plots < self.n_attn_plots:
                self.plot_attn(attns, 'inf_ed')
                self.sampled_attention_plots += 1

            ref_audio_name = self.reference_names[i]
            raw_path = os.path.join(self.datadir, '16k_wav', ref_audio_name)
            shutil.copyfile(raw_path + '.wav', os.path.join(self.sampledir, f'epoch{self.current_epoch}-{i}-ref.wav'))
