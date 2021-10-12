from .FeatureFuser import Wav2vec2Wrapper
import pytorch_lightning.core.lightning as pl

class MinimalClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2vec2Wrapper(pretrain=False)

    def forward(self, x, length=None):
        reps = self.wav2vec2(x, length)
        return reps
