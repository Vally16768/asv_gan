# detector_wrapper.py
import torch, torchaudio
class MyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ...  # încarcă greutățile tale
    def forward(self, x):
        """
        Acceptă fie waveform [B, L], fie mel [B, n_mels, T].
        Returnează score scalar [B] (prob spoof sau logit).
        """
        if x.dim() == 2:   # waveform
            # extrage feature intern și apoi model(...)
            pass
        else:              # mel
            pass
        return score.squeeze(-1)
