
import torch
from pathlib import Path
from constants import SR, SAVE_DIR, CKPT_DIR, SAMPLES_DIR
from features import LogMel
from models import Generator
from utils import save_wave
from detector_wrapper import DetectorWrapper

@torch.no_grad()
def load_ckpt(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing):
        print("Missing keys:", missing[:5], "...")
    if len(unexpected):
        print("Unexpected keys:", unexpected[:5], "...")

def evaluate_sample(ckpt="best.pth", wav: torch.Tensor = None, keras_loader_fn=None):
    G = Generator().eval()
    load_ckpt(G, Path(CKPT_DIR) / ckpt)

    if wav is None:
        raise RuntimeError("Provide a waveform tensor [T]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = G.to(device)
    x = wav.unsqueeze(0).to(device)

    y = G(x).squeeze(0).cpu()
    # Save
    outp = Path(SAMPLES_DIR) / "eval_out.wav"
    save_wave(outp, y, SR)

    # Keras score
    det = DetectorWrapper(keras_loader_fn=keras_loader_fn)
    p = det.keras_prob(y.unsqueeze(0))
    print("ASV bona_fide prob:", float(p.mean()))
    return y, float(p.mean())
