import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import (DecoderPrenet, EmotionPrenet, Postnet, PositionEmbedding,
                     TransformerDecoderGuided, TransformerDecoderLayer,
                     TransformerEncoder, TransformerEncoderLayer, EmoDecoderLayer, PairedTransformerDecoderGuided, PairedDecoderLayer,
                     PairedTransformerEncoderLayer, PairedTransformerEncoder)
from g2p_en import G2p
import random
import numpy as np

class ETTSTransformer_baseline(nn.Module):
    def __init__(self, text_embed_dim, emo_embed_dim, nmels, maxlength, symbols, ngst,
                 d_model=512, hidden_size=1536, nlayers=4, nheads=4):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.text_embedding = nn.Embedding(len(symbols.keys()), text_embed_dim, padding_idx=symbols['<pad>'])
        self.global_style_embedding = nn.Embedding(ngst, d_model)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nheads, hidden_size),
            nlayers
        )
        self.decoder = TransformerDecoderGuided(
            TransformerDecoderLayer(d_model, nheads, hidden_size),
            nlayers
        )
        self.emotion_global_prenet = EmotionPrenet(emo_embed_dim, hidden_size=d_model, dropout=0.1)
        self.emotion_global_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.text_norm = nn.LayerNorm(text_embed_dim)
        self.decoder_prenet = DecoderPrenet(nmels, bottleneck_size=d_model//8, hidden_size=d_model)
        self.mel_linear = nn.Linear(d_model, nmels)
        self.postnet = Postnet(nmels, hidden_size=d_model, dropout=0.1)
        self.stop_pred = nn.Linear(d_model, 1)

        self.pos_txt = PositionEmbedding(maxlength, d_model, scaled=False)
        self.pos_mel = PositionEmbedding(maxlength, d_model, scaled=False)
        tgt_mask = (torch.tril(torch.ones(maxlength, maxlength), diagonal=0) == 0)
        self.register_buffer('tgt_mask', tgt_mask)

        self.maxlength = maxlength
        self.nmels = nmels

    def length2mask(self, x, length):
        mask = torch.zeros(
            x.shape[:2], dtype=x.dtype, device=x.device
        )
        mask[
            (torch.arange(mask.shape[0], device=x.device), length - 1)
        ] = 1
        mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return ~mask

    def forward(self, emotion_global, text, mels, length_emo_global, length_text, length_mels):
        text = self.pos_txt(self.text_norm(self.text_embedding(text)))
        txt_mask = self.length2mask(text, length_text)
        text = text.transpose(0, 1) #T, B, C
        text, enc_gloss, enc_attn = self.encoder(text, src_key_padding_mask=txt_mask)
        src, src_padding_mask = text, txt_mask
        emo_mask_global = self.length2mask(emotion_global, length_emo_global)
        emotion_global = self.emotion_global_prenet(emotion_global, mask=emo_mask_global) #(B,T,C)
        keys = torch.tanh(self.global_style_embedding.weight).unsqueeze(0).expand(emotion_global.size(0), -1, -1)
        emotion_global = self.emotion_global_attn(emotion_global, keys, keys)[0].transpose(0, 1)
        src = emotion_global.expand(src.size(0), -1, -1) + src

        mels = self.pos_mel(self.decoder_prenet(mels, dropout=0.2))
        mel_padding_mask = self.length2mask(mels, length_mels)
        mels = mels.transpose(0, 1) #T, B, C
        mels = emotion_global.expand(mels.size(0), -1, -1) + mels
        mels = torch.cat([torch.zeros((1, mels.size(1), mels.size(2)), device=mels.device), mels[: -1]], 0) #shift
        tgt_len = mels.size(0)
        tgt_mask = self.tgt_mask[: tgt_len, : tgt_len]

        output, ed_gloss, attn, dec_attn = self.decoder(mels, src,
                                                        tgt_mask=tgt_mask,
                                                        tgt_key_padding_mask=mel_padding_mask,
                                                        text_key_padding_mask=src_padding_mask) #TBC
        gloss = ed_gloss + enc_gloss
        gate_logit = self.stop_pred(output).squeeze(-1).transpose(0, 1)
        output = output.transpose(0, 1) #B, T, C
        mel_out = self.mel_linear(output)
        mel_out_post = mel_out.detach() + self.postnet(mel_out.detach())

        return mel_out, mel_out_post, gate_logit, mel_padding_mask, gloss, attn, enc_attn, dec_attn


    def inference(self, emotion_global, text, maxlen=1000, threshold=.5, ref_wav=True):
        text = self.pos_txt(self.text_norm(self.text_embedding(text)))
        text = text.transpose(0, 1) #T, B, C
        mels = torch.zeros((1, 1, self.nmels), device=text.device) #B, T, C, B=1
        attn_mask = torch.zeros((self.nheads, 1, text.size(0)), device=text.device)
        text, _, _ = self.encoder(text)
        src = text
        emotion_global = self.emotion_global_prenet(emotion_global)
        keys = torch.tanh(self.global_style_embedding.weight).unsqueeze(0)
        emotion_global = self.emotion_global_attn(emotion_global, keys, keys)[0].transpose(0, 1)
        src = emotion_global.expand(src.size(0), -1, -1) + src
        start_token = torch.zeros((1, 1, self.d_model), device=text.device)
        #Decode
        for i in range(maxlen):
            if i == 0:
                mels_pos = start_token
            else:
                mels_pos = self.pos_mel(self.decoder_prenet(mels[:, 1:], dropout=0.2)).transpose(0, 1)
                mels_pos = emotion_global.expand(mels_pos.size(0), -1, -1) + mels_pos
                mels_pos = torch.cat([start_token, mels_pos], 0)
            output, _, attns, _ = self.decoder(mels_pos, src)#, memory_mask=attn_mask)
            gate_logit = self.stop_pred(output[-1]).item()
            output = output[-1] #B=1, C
            mel_out = self.mel_linear(output)
            mels = torch.cat([mels, mel_out.unsqueeze(1)], 1)
            if gate_logit > threshold:
                break
        mels = mels[:, 1:]
        mels = mels + self.postnet(mels)
        return mels, attns
