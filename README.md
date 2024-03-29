# LST-TTS
Official implementation for the paper [Fine-grained style control in transformer-based text-to-speech synthesis](http://arxiv.org/abs/2110.06306).
Submitted to ICASSP 2022.
**Audio samples/demo for our system can be accessed [here](https://b04901014.github.io/FG-transformer-TTS/)**

 - Mar. 5 2022: Fixed a inference bug of not passing the causal mask, quality of samples should be slightly better. (I have not updated the demos with this.)

## Setting up submodules
```
git submodule update --init --recursive
```
Get the waveglow vocoder checkpoint from [here](https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF) (This is from the [NVIDIA official WaveGlow repo](https://github.com/NVIDIA/waveglow)).

## Setup environment
See `docker/Dockerfile` for the packages need to be installed.

## Dataset preprocessing
### [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
```
python preprocess_LJSpeech.py --datadir LJSpeechDir --outputdir OutputDir
```
### [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)
Get the leading and trailing scilence marks from [this repo](https://github.com/nii-yamagishilab/vctk-silence-labels), and put `vctk-silences.0.92.txt` in your VCTK dataset directory.
```
python preprocess_VCTK.py --datadir VCTKDir --outputdir Output_Train_Dir
```
```
python preprocess_VCTK.py --datadir VCTKDir --outputdir Output_Test_Dir --make_test_set
```
 - `--make_test_set`: specify this flag to process the speakers in the test set, otherwise only process training speakers.
## Training
### LJSpeech
```
python train_TTS.py --precision 16 \
                    --datadir FeatureDir \
                    --vocoder_ckpt_path WaveGlowCKPT_PATH \
                    --sampledir SampleDir \
                    --batch_size 128 \
                    --check_val_every_n_epoch 50 \
                    --use_guided_attn \
                    --training_step 250000 \
                    --n_guided_steps 250000 \
                    --saving_path Output_CKPT_DIR \
                    --datatype LJSpeech \
                    [--distributed]
```
 - `--distributed`: enable DDP multi-GPU training
 - `--batch_size`: batch size **per GPU**, scale down if you train with multi-GPU and want to keep the same batch size
 - `--check_val_every_n_epoch`: sample and validate every n epoch
 - `--datadir`: output directory of the preprocess scripts
### VCTK
```
python train_TTS.py --precision 16 \
                    --datadir FeatureDir \
                    --vocoder_ckpt_path WaveGlowCKPT_PATH \
                    --sampledir SampleDir \
                    --batch_size 64 \
                    --check_val_every_n_epoch 50 \
                    --use_guided_attn \
                    --training_step 150000 \
                    --n_guided_steps 150000 \
                    --etts_checkpoint LJSpeech_Model_CKPT \
                    --saving_path Output_CKPT_DIR \
                    --datatype VCTK \
                    [--distributed]
```
 - `--etts_checkpoint`: the checkpoint path of pretrained model (on LJ Speech)

### Synthesis
We provide examples for synthesis of the system in `synthesis.py`, you can adjust this script to your own usage.
Example to run `synthesis.py`:
```
python synthesis.py --etts_checkpoint VCTK_Model_CKPT \
                    --sampledir SampleDir \
                    --datatype VCTK \
                    --vocoder_ckpt_path WaveGlowCKPT_PATH
```

### Pretrained checkpoints
We provide pretrained checkpoints on LJ Speech and VCTK. The model is a little large since it contains all the training and optimizer states.
 - [LJ Speech](https://cmu.box.com/s/wjwx9vwnhe76hetxawjf6d2r7as3yvxa)
 - [VCTK](https://cmu.box.com/s/bu7fo2mk709sl064718o2jk3nsvbo3ix)
