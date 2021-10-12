## Fine-grained Style Control in Transformer-based Text-to-speech Synthesis

 - Submitted to ICASSP 2022
 - In this paper, we present a novel architecture to realize fine-grained style control on the transformer-based text-to-speech synthesis (TransformerTTS). Specifically, we model the speaking style by extracting a time sequence of local style tokens (LST) from the reference speech. The existing content encoder in TransformerTTS is then replaced by our designed cross-attention blocks for fusion and alignment between content and style. As the fusion is performed along with the skip connection, our cross-attention block provides a good inductive bias to gradually infuse the phoneme representation with a given style. Additionally, we prevent the style embedding from encoding linguistic content by randomly truncating LST during training and using wav2vec 2.0 features. Experiments show that with fine-grained style control, our system performs better in terms of naturalness, intelligibility, and style transferability. Our code and samples are publicly available.
 - [Code](...)
 - [Paper](...)

### LJ Speech samples (single speaker)

|Style reference speech|Generated Speech|Text|
|----------------------|----------------|----|
|<audio src="samples/LJSpeech/1_ref.wav" type="audio/wav"  controls preload></audio>|<audio src="samples/LJSpeech/1_ref.wav" type="audio/wav"  controls preload></audio>|<embed src="samples/LJSpeech/1.txt" style="width:100%; height:100%; margin: 0 auto;">|
