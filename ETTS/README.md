# Emotional Text to Speech Synthesis (In development)
## Setting up submodules
```
git submodule update --init --recursive
```
Get the waveglow vocoder checkpoint from [here](https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF).

## Dataset preprocessing
We use two datasets in our training pipeline:
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): First phase training for the attention alignments
```
python preprocess_LJSpeech.py --datadir LJSpeechDir --outputdir OutputDir --emo_model_dir PretrainedModelDir
```
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): Training with multiple speakers to learn speaker/emotion embeddings
```
python preprocess_LibriTTS.py --datadir LibriTTSDir --outputdir OutputDir --emo_model_dir PretrainedModelDir
```
## Training
```
python run_downstream_etts.py --precision 16 \
                              --datadir DatasetDir \
                              --vocoder_ckpt_path WaveGlowCkpt \
                              --sampledir SampleDir \
                              --use_guided_attn \
                              --distributed
```
