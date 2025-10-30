# eval.py — evaluare aliniată cu antrenarea (normalizare + potrivire RMS), preferă G_EMA
import argparse
from pathlib import Path
import torch

from constants import SR, CKPT_DIR, SAMPLES_DIR
from models import Generator
from utils import save_wave

try:
    from detector_wrapper import DetectorWrapper  # opțional
except Exception:
    DetectorWrapper = None

@torch.no_grad()
def _rms_per_sample(w, eps: float = 1e-8):
    if w.ndim == 1:
        w = w.unsqueeze(0)
    return (w.float().pow(2).mean(dim=1, keepdim=True) + eps).sqrt()

@torch.no_grad()
def _norm_wave_per_sample(x, eps: float = 1e-6):
    if x.ndim == 1:
        x = x.unsqueeze(0)
    mean = x.mean(dim=1, keepdim=True)
    std  = x.std(dim=1, keepdim=True)
    std  = torch.where(std < eps, torch.full_like(std, eps), std)
    xn   = (x - mean) / std
    return xn.clamp(-3.0, 3.0)

@torch.no_grad()
def _match_rms_for_saving(y, x_ref):
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if x_ref.ndim == 1:
        x_ref = x_ref.unsqueeze(0)
    ry = _rms_per_sample(y)
    rx = _rms_per_sample(x_ref)
    # evaluare: mai permisiv ca să fie audibil în debugging
    scale = (rx / ry.clamp(min=1e-6)).clamp(0.25, 64.0)
    return (y * scale).clamp(-1.0, 1.0)

@torch.no_grad()
def _load_ckpt_into(model: torch.nn.Module, ckpt_path: Path):
    sd = torch.load(ckpt_path, map_location="cpu")
    # suport state dict împachetat
    if isinstance(sd, dict) and ("state_dict" in sd):
        sd = sd["state_dict"]
    # preferă G_EMA
    if isinstance(sd, dict) and ("G_EMA" in sd):
        sd = sd["G_EMA"]
    elif isinstance(sd, dict) and ("G" in sd):
        sd = sd["G"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected

@torch.no_grad()
def run_eval(wav: torch.Tensor, ckpt: str = "best.pth", device: str | None = None, keras_loader_fn=None):
    if wav is None or wav.numel() == 0:
        raise RuntimeError("Provide a waveform tensor [T] or [1,T]")
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    G = Generator().to(device).eval()
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        ckpt_path = Path(CKPT_DIR) / ckpt
    missing, unexpected = _load_ckpt_into(G, ckpt_path)
    if missing or unexpected:
        print(f"State dict missing={len(missing)} unexpected={len(unexpected)}")

    x_raw = wav.to(device)                 # [1,T] brut
    x     = _norm_wave_per_sample(x_raw)   # *** aliniere cu train ***
    y     = G(x).squeeze(0).detach().cpu() # [T]
    y_out = _match_rms_for_saving(y, wav.squeeze(0))

    outp = Path(SAMPLES_DIR) / "eval_out.wav"
    save_wave(outp, y_out, SR)
    print(f"Saved: {outp}")

    p_bona = None
    if DetectorWrapper is not None:
        try:
            det = DetectorWrapper(keras_loader_fn=keras_loader_fn)
            p_bona = float(det.keras_prob(y_out.unsqueeze(0)).mean())
            print("ASV bona_fide prob:", p_bona)
        except Exception as e:
            print("DetectorWrapper failed:", e)

    return y_out, p_bona

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Cale către fișierul .wav de evaluat")
    parser.add_argument("--ckpt", type=str, default="best.pth")
    args = parser.parse_args()

    import soundfile as sf
    import numpy as np
    w, sr = sf.read(args.wav)
    assert sr == SR, f"Sample rate mismatch: got {sr}, expected {SR}"
    if w.ndim > 1:
        w = np.mean(w, axis=1)
    w = torch.from_numpy(w).float()
    run_eval(w, ckpt=args.ckpt)

if __name__ == "__main__":
    cli()
