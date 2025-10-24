# detector_wrapper.py
import torch
class DetectorWrapper(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = torch.load(model_path, map_location=device)
        self.model.to(device)
        self.model.eval()
    def forward(self, wave):
        # ensure dims [B,L] -> your model may apply feature extraction internally
        return self.model(wave)
