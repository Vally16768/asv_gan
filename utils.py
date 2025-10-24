# utils.py
import torch
import torchaudio
import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80

mel_to_spec_inv = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_iter=32
)

def mel_to_wave(mel_log):
    # mel_log is log1p(mel) in dataset; we invert: mel = exp(mel_log) - 1
    mel = torch.expm1(mel_log)
    # Need to convert mel spectrogram back to linear STFT magnitude:
    # torchaudio doesn't have inverse mel matrix by default; construct via librosa / pseudo-inverse
    import librosa
    mel_np = mel.detach().cpu().numpy()
    # mel_np shape [B, n_mels, T]
    waves = []
    inv_mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    inv_mel = np.linalg.pinv(inv_mel)
    for m in mel_np:
        S = np.maximum(1e-8, inv_mel @ m)  # linear mag
        # Griffin-Lim expects magnitude stft shape [freq_bins, time]
        # but torchaudio's GriffinLim works on stft complex? We'll use librosa griffinlim for simplicity:
        wav = librosa.griffinlim(S, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_iter=32)
        waves.append(wav)
    # return tensor [B, L]
    max_len = max([len(w) for w in waves])
    out = np.zeros((len(waves), max_len), dtype=np.float32)
    for i,w in enumerate(waves): out[i,:len(w)] = w
    return torch.from_numpy(out)

def save_wav(path, wav, sr=SR):
    # wav: numpy or torch 1D
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    sf.write(path, wav, sr)

def load_detector(model_path, device):
    """
    Expected: detector is a PyTorch model saved with torch.save(model.state_dict()) OR the whole model.
    The detector must provide a function: detector(waveform_tensor) -> score tensor [B]
    Here we try to load model and provide wrapper. If your detector code is custom, replace loader.
    """
    import torch
    try:
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print("Failed to auto-load detector. Please provide a wrapper that loads & returns detector model.")
        raise e
