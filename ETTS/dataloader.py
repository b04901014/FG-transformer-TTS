import os
from torch.utils import data
import torch
import json
import numpy as np
from g2p_en import G2p
import soundfile as sf

class PreprocessedDataset(data.Dataset):
    def __init__(self, datadir):
        self.meldir = os.path.join(datadir, 'mels')
        self.emodir = os.path.join(datadir, 'emo_reps')
        self._labelset = G2p().phonemes
        self._labelset += [',', '.', ':', ';', '?', '!', '-', '\"', '\'', ' ']
        self.labelset = {x: i for i, x in enumerate(self._labelset)}
        with open(os.path.join(datadir, 'metadata.json'), 'r') as f:
            self.label = json.load(f) #{wavname: {length: l, text: text, phonemes: label}}
        self.audio_names, self.lengths = [], []
        for k, v in self.label.items():
            self.audio_names.append(k)
            self.lengths.append(v['length'])

        #Print statistics:
        l = len(self.audio_names)
        print (f'Total {l} examples')

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, i):
        dataname = self.audio_names[i]
        emos = np.load(os.path.join(self.emodir, dataname + '.npy'))
        _phonemes = self.label[dataname]['phonemes']
        phonemes = [self.labelset['<s>']]
        for i, phoneme in enumerate(_phonemes):
            if phoneme in self.labelset:
                phonemes.append(self.labelset[phoneme])
            else:
                phonemes.append(self.labelset['<unk>'])
        phonemes.append(self.labelset['</s>'])
        mels = np.load(os.path.join(self.meldir, dataname + '.npy'))
        return mels.T, emos, phonemes, dataname


class LJSpeechDataset(PreprocessedDataset):
    def seqCollate(self, batch):
        l_mel, l_emo, l_phoneme, names = [], [], [], []
        for b in batch:
            mel, emo, phoneme, name = b
            names.append(name)
            l_mel.append(len(mel))
            l_emo.append(len(emo))
            l_phoneme.append(len(phoneme))
        ml_mel, ml_emo, ml_phoneme = max(l_mel), max(l_emo), max(l_phoneme)
        mels, emos, phonemes = [], [], []
        for b in batch:
            mel, emo, phoneme = b[: 3]
            mels.append(np.pad(mel, [[0, ml_mel - len(mel)], [0, 0]]))
            emos.append(np.pad(emo, [[0, ml_emo - len(emo)], [0, 0]]))
            phonemes.append(np.pad(phoneme, [[0, ml_phoneme - len(phoneme)]], constant_values=self.labelset['<pad>']))
        ret = [torch.FloatTensor(mels), torch.LongTensor(l_mel),
               torch.FloatTensor(emos), torch.LongTensor(l_emo),
               torch.LongTensor(phonemes), torch.LongTensor(l_phoneme)]
        ret.append(names)
        return ret

class VCTKDataset(PreprocessedDataset):
    def __init__(self, datadir):
        super().__init__(datadir)
        self.speaker_groups = dict()
        for name in self.audio_names:
            speakerid = name.split('_')[0]
            if speakerid in self.speaker_groups:
                self.speaker_groups[speakerid].append(name)
            else:
                self.speaker_groups[speakerid] = [name]

    def __getitem__(self, i):
        mels, emos, phonemes, name = super().__getitem__(i)
        speakerid = name.split('_')[0]
        speaker_group = self.speaker_groups[speakerid]
        random_utter = np.random.choice(speaker_group)
        same_spkr_emos = np.load(os.path.join(self.emodir, random_utter + '.npy'))
        return mels, emos, same_spkr_emos, phonemes, name

    def seqCollate(self, batch):
        l_mel, l_emo, l_same_emo, l_phoneme, names = [], [], [], [], []
        for b in batch:
            mel, emo, same_spkr_emo, phoneme, name = b
            names.append(name)
            l_mel.append(len(mel))
            l_emo.append(len(emo))
            l_same_emo.append(len(same_spkr_emo))
            l_phoneme.append(len(phoneme))
        ml_mel, ml_emo, ml_same_emo, ml_phoneme = max(l_mel), max(l_emo), max(l_same_emo), max(l_phoneme)
        mels, emos, same_spkr_emos, phonemes = [], [], [], []
        for b in batch:
            mel, emo, same_spkr_emo, phoneme, _ = b
            mels.append(np.pad(mel, [[0, ml_mel - len(mel)], [0, 0]]))
            emos.append(np.pad(emo, [[0, ml_emo - len(emo)], [0, 0]]))
            same_spkr_emos.append(np.pad(same_spkr_emo, [[0, ml_same_emo - len(same_spkr_emo)], [0, 0]]))
            phonemes.append(np.pad(phoneme, [[0, ml_phoneme - len(phoneme)]], constant_values=self.labelset['<pad>']))
        ret = [torch.FloatTensor(mels), torch.LongTensor(l_mel),
               torch.FloatTensor(emos), torch.LongTensor(l_emo),
               torch.FloatTensor(same_spkr_emos), torch.LongTensor(l_same_emo),
               torch.LongTensor(phonemes), torch.LongTensor(l_phoneme)]
        ret.append(names)
        return ret

def RandomBucketSampler(nbuckets, length, batch_size, drop_last, distributed=False,
                        world_size=None, rank=None):
    if distributed:
        return DistributedRandomBucketSampler(nbuckets, length, batch_size, drop_last, world_size, rank)
    return SingleRandomBucketSampler(nbuckets, length, batch_size, drop_last)

class SingleRandomBucketSampler(data.Sampler):
    def __init__(self, nbuckets, length, batch_size, drop_last):
        self.length = length
        self.batch_size = batch_size
        self.drop_last = drop_last
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])

    def __iter__(self):
        random.shuffle(self.indices)
        for x in self.indices:
            random.shuffle(x)
        idxs = [i for x in self.indices for i in x]
        batches, batch, sum_len = [], [], 0
        for idx in idxs:
            batch.append(idx)
            sum_len += 1
            if sum_len >= self.batch_size:
                batches.append(batch)
                batch, sum_len = [], 0
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.length) // self.batch_size  # type: ignore
        else:
            return (len(self.length) + self.batch_size - 1) // self.batch_size

class DistributedRandomBucketSampler(data.Sampler):
    def __init__(self, nbuckets, length, batch_size, drop_last, num_replicas, rank, seed=787):
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.length = length
        self.batch_size = batch_size
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        if self.drop_last and len(length) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(length) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(length) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        #Deterministic shuffling
        random.Random(self.epoch + self.seed).shuffle(self.indices)
        for i, x in enumerate(self.indices):
            seed = self.epoch + self.seed + i * 5
            random.Random(seed).shuffle(x)
        indices = [i for x in self.indices for i in x]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank*self.num_samples: (self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        #Batching
        batches, batch = [], []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        #Stochastic suffling
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size  # type: ignore
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
