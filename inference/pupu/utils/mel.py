import torch
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    # Min value: ln(1e-5) = -11.5129
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def extract_mel_features(
    y,
    cfg,
    center=False,
):
    global mel_basis, hann_window
    if cfg.fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mel,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        mel_basis[str(cfg.fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(cfg.fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec.squeeze(0)
