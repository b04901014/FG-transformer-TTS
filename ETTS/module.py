import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class EmotionPrenet(nn.Module):
    def __init__(self, ndim, hidden_size=256, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(ndim, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        #BTC -> B1C
        x = self.dropout(x)
        x = F.relu(self.linear(x))
        x, _ = self.rnn(x)
        if mask is not None:
            mask = (~mask).float().unsqueeze(-1)
            x = x * mask
            x = x.sum(1) / mask.sum(1)
        else:
            x = x.mean(1)
        return x.unsqueeze(1)

class DecoderPrenet(nn.Module):
    def __init__(self, nmels, bottleneck_size=32, hidden_size=256):
        super().__init__()
        self.l1 = nn.Linear(nmels, bottleneck_size)
        self.l2 = nn.Linear(bottleneck_size, bottleneck_size)
        self.l3 = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x, dropout):
        x = F.relu(self.l1(x))
        x = F.dropout(x, p=dropout, training=True) #Inference also dropout
        x = F.relu(self.l2(x))
        x = F.dropout(x, p=dropout, training=True)
        x = F.relu(self.l3(x))
        return x

#Adapted from NVIDIA/tacotron2
class Postnet(nn.Module):
    def __init__(self, nmels, hidden_size=512, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv = nn.ModuleList()
        self.conv.append(
            nn.Sequential(
                nn.Conv1d(nmels, hidden_size, 5, 1, 2),
                nn.BatchNorm1d(hidden_size)
            )
        )
        for _ in range(3):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 5, 1, 2),
                    nn.BatchNorm1d(hidden_size)
                )
            )
        self.final = nn.Sequential(
            nn.Conv1d(hidden_size, nmels, 5, 1, 2),
            nn.BatchNorm1d(nmels)
        )

    def forward(self, x):
        #(B, T, C) -> (B, T, C)
        x = x.transpose(1, 2)
        for conv in self.conv:
            x = F.dropout(torch.tanh(conv(x)), p=self.dropout, training=self.training)
        x = F.dropout(self.final(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, maxlength, ndim, scaled=True):
        super().__init__()
        p = torch.zeros((maxlength, ndim))
        pi = torch.arange(start=0, end=maxlength).float().unsqueeze(1)
        pi = pi * torch.exp(torch.arange(start=0, end=ndim, step=2).float() * -(np.log(10000.0) / ndim))
        p[:, 0::2] = torch.sin(pi)
        p[:, 1::2] = torch.cos(pi)
        self.register_buffer('p', p)
        self.scaled = scaled
        self.scalar = nn.Parameter(torch.FloatTensor([1.0])) if scaled else 1.0


    def forward(self, x):
        B, L, C = x.size()
        p = self.p[: L].unsqueeze(0).expand(B, -1, -1)
        x = x + self.scalar * p
        return x

def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(
        -((grid_y.float() / rolen - grid_x.float() / rilen) ** 2) / (2 * (sigma ** 2))
    )

class EmoDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class PairedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#        self.multihead_attn_emo = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_joint = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, text, emotion, tgt_mask=None, text_mask=None, emo_mask=None,
                tgt_key_padding_mask=None, text_key_padding_mask=None, emo_key_padding_mask=None):
        tgt2, self_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2_joint, attn_joint = self.multihead_attn_joint(text, emotion, emotion, attn_mask=emo_mask,
                                                          key_padding_mask=emo_key_padding_mask)
        text_emo = text + self.dropout4(tgt2_joint)
        text_emo = self.norm4(text_emo)
        tgt2_text, attn = self.multihead_attn_text(tgt, text_emo, text_emo, attn_mask=text_mask,
                                                   key_padding_mask=text_key_padding_mask)
#        tgt2_emo, attn_emo = self.multihead_attn_emo(tgt2_text, emotion, emotion, attn_mask=emo_mask,
#                                                     key_padding_mask=emo_key_padding_mask)
        tgt2 = tgt2_text
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn, self_attn, attn_joint

#Adjusted from pytorch
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, self_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn, self_attn

class TransformerDecoderGuided(nn.Module):
    def __init__(self, decoder_layers,
                 guided_sigma=0.7, guided_layers=None,
                 norm=None):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.num_layers = len(decoder_layers)
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else self.num_layers

    def forward(self, tgt, text, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                text_key_padding_mask=None):#, emotion_key_padding_mask=None):
        output = tgt
        guided_loss = 0

        attns = []
        self_attns = []
        memory_key_padding_mask = text_key_padding_mask
        memory = text
        for i, mod in enumerate(self.layers):
            output, attn, self_attn = mod(output, memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn.detach())
            self_attns.append(self_attn.detach())
            if i < self.guided_layers and tgt_key_padding_mask is not None:
                t_length = (~tgt_key_padding_mask).float().sum(-1)
                s_length = (~text_key_padding_mask).float().sum(-1)
                attn_w = _make_guided_attention_mask(tgt_key_padding_mask.size(-1), t_length, text_key_padding_mask.size(-1), s_length, self.guided_sigma)

                g_loss = attn * attn_w #N, L, S
                non_padding_mask = (~tgt_key_padding_mask).unsqueeze(-1) & (~memory_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, attns, self_attns

class PairedTransformerDecoderGuided(nn.Module):
    def __init__(self, decoder_layer, num_layers, d_model, nhead,
                 guided_sigma=0.5, guided_layers=None,
                 norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else num_layers

        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, tgt, text, global_emotion, local_emotion, tgt_mask=None,
                tgt_key_padding_mask=None, text_key_padding_mask=None, emotion_key_padding_mask=None):
        output = tgt
        guided_loss = 0

        attns = []
        self_attns = []
        local_emotion = global_emotion.expand(local_emotion.size(0), -1, -1) + local_emotion
        tgt2_joint, attn_emo = self.multihead_attn1(text, local_emotion, local_emotion,
                                                    key_padding_mask=emotion_key_padding_mask)
        text_emo = text + self.dropout1(tgt2_joint)
        text_emo = self.norm1(text_emo)
        for i, mod in enumerate(self.layers):
            output, attn, self_attn = mod(output, text_emo, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=text_key_padding_mask)
            attns.append(attn.detach())
            self_attns.append(self_attn.detach())
            if i < self.guided_layers and tgt_key_padding_mask is not None:
                t_length = (~tgt_key_padding_mask).float().sum(-1)
                s_length = (~text_key_padding_mask).float().sum(-1)
                emo_length = (~emotion_key_padding_mask).float().sum(-1)
                attn_w = _make_guided_attention_mask(tgt_key_padding_mask.size(-1), t_length, text_key_padding_mask.size(-1), s_length, self.guided_sigma)
                attn_w_emo = _make_guided_attention_mask(text_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)

                g_loss = attn * attn_w #N, L, S
                non_padding_mask = (~tgt_key_padding_mask).unsqueeze(-1) & (~text_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss[non_padding_mask].mean() + guided_loss
                g_loss_emo = attn_emo * attn_w_emo #N, L, S
                non_padding_mask = (~text_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, attns, self_attns

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layers,
                 guided_sigma=0.3, guided_layers=None,
                 norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.num_layers = len(encoder_layers)
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else self.num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        self_attns = []
        guided_loss = 0
        for i, mod in enumerate(self.layers):
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            self_attns.append(attn.detach())
            if i < self.guided_layers and src_key_padding_mask is not None:
                t_length = (~src_key_padding_mask).float().sum(-1)
                s_length = (~src_key_padding_mask).float().sum(-1)
                attn_w = _make_guided_attention_mask(attn.size(1), t_length, attn.size(2), s_length, self.guided_sigma)
                g_loss = attn * attn_w #N, L, S
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~src_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, self_attns

class PairedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(PairedTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.multihead_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, src, global_emotion, local_emotion,
                src_mask=None, src_key_padding_mask=None, emotion_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        local_emotion = global_emotion.expand(local_emotion.size(0), -1, -1) + local_emotion
        src2, attn_emo = self.multihead_attn3(src, local_emotion, local_emotion,
                                              key_padding_mask=emotion_key_padding_mask)
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        src2 = self.linear4(self.dropout(self.activation(self.linear3(src))))
        src = src + self.dropout4(src2)
        src = self.norm4(src)
        return src, attn, attn_emo

class PairedTransformerEncoder(nn.Module):
    def __init__(self, encoder_layers,
                 guided_sigma=0.3, guided_layers=None,
                 norm=None):
        super(PairedTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.num_layers = len(self.layers)
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else self.num_layers

    def forward(self, src, global_emotion, local_emotion,
                mask=None, src_key_padding_mask=None, emotion_key_padding_mask=None):
        output = src
        self_attns = []
        guided_loss = 0
        for i, mod in enumerate(self.layers):
            output, attn, attn_emo = mod(output, global_emotion, local_emotion,
                                         src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                                         emotion_key_padding_mask=emotion_key_padding_mask)
            self_attns.append(attn.detach())
            if i < self.guided_layers and src_key_padding_mask is not None:
                s_length = (~src_key_padding_mask).float().sum(-1)
                attn_w = _make_guided_attention_mask(attn.size(1), s_length, attn.size(2), s_length, self.guided_sigma)
                g_loss = attn * attn_w #N, L, S
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~src_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss[non_padding_mask].mean() + guided_loss

                emo_length = (~emotion_key_padding_mask).float().sum(-1)
                attn_w_emo = _make_guided_attention_mask(src_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)

                g_loss_emo = attn_emo * attn_w_emo #N, L, S
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, self_attns

