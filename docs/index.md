## Fine-grained Style Control in Transformer-based Text-to-speech Synthesis

 - Submitted to ICASSP 2022
 - In this paper, we present a novel architecture to realize fine-grained style control on the transformer-based text-to-speech synthesis (TransformerTTS). Specifically, we model the speaking style by extracting a time sequence of local style tokens (LST) from the reference speech. The existing content encoder in TransformerTTS is then replaced by our designed cross-attention blocks for fusion and alignment between content and style. As the fusion is performed along with the skip connection, our cross-attention block provides a good inductive bias to gradually infuse the phoneme representation with a given style. Additionally, we prevent the style embedding from encoding linguistic content by randomly truncating LST during training and using wav2vec 2.0 features. Experiments show that with fine-grained style control, our system performs better in terms of naturalness, intelligibility, and style transferability. Our code and samples are publicly available.
 - [Code](...)
 - [Paper](...)

### LJ Speech samples (single speaker)

|Style reference speech|Generated Speech|Text|
|----------------------|----------------|----|
|<audio src="samples/LJSpeech/1_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/1_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/1.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/2_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/2_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/2.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/3_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/3_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/3.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/4_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/4_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/4.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/5_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/5_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/5.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/6_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/6_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/6.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/7_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/7_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/7.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/8_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/8_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/8.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/9_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/9_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/9.txt" width="300" height="80">|
|<audio src="samples/LJSpeech/10_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/10_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/10.txt" width="300" height="80">|

### VCTK (multiple speakers, samples from speakers held-out in the test set)

#### Same style and speaker reference speech

|Style reference speech|Generated Speech|Text|
|----------------------|----------------|----|
|<audio src="samples/VCTK_align/1_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_align/1_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_align/1.txt" width="300" height="80">|
|<audio src="samples/VCTK_align/2_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_align/2_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_align/2.txt" width="300" height="80">|
|<audio src="samples/VCTK_align/3_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_align/3_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_align/3.txt" width="300" height="80">|
|<audio src="samples/VCTK_align/4_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_align/4_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_align/4.txt" width="300" height="80">|
|<audio src="samples/VCTK_align/5_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_align/5_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_align/5.txt" width="300" height="80">|

#### Different style and speaker reference speech

|Speaker reference speech|Style reference speech|Generated Speech|Text|
|------------------------|----------------------|----------------|----|
|<audio src="samples/VCTK_diff/1_ref_global.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/1_ref_local.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/1_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_diff/1.txt" width="300" height="80">|
|<audio src="samples/VCTK_diff/2_ref_global.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/2_ref_local.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/2_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_diff/2.txt" width="300" height="80">|
|<audio src="samples/VCTK_diff/3_ref_global.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/3_ref_local.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/3_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_diff/3.txt" width="300" height="80">|
|<audio src="samples/VCTK_diff/4_ref_global.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/4_ref_local.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/4_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_diff/4.txt" width="300" height="80">|
|<audio src="samples/VCTK_diff/5_ref_global.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/5_ref_local.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/VCTK_diff/5_syn.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/VCTK_diff/5.txt" width="300" height="80">|

### Observations
 - Still are some minor deletion or repetition TTS errors
 - Samples for male speakers in VCTK are significantly lower in quality due to the use of LJ Speech as pre-training data
 - Distortion occurs often during high loudness, pitch transitions.
