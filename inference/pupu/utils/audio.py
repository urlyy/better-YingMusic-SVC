import torch
import numpy as np
import librosa
import soundfile as sf
import torchaudio


def load_audio_torch(wave_file, fs):
    """Load audio data into torch tensor

    Args:
        wave_file (str): path to wave file
        fs (int): sample rate

    Returns:
        audio (tensor): audio data in tensor
        fs (int): sample rate
    """

    audio, sample_rate = librosa.load(wave_file, sr=fs, mono=True)
    # audio: (T,)
    assert len(audio) > 2

    # Check the audio type (for soundfile loading backbone) - float, 8bit or 16bit
    if np.issubdtype(audio.dtype, np.integer):
        max_mag = -np.iinfo(audio.dtype).min
    else:
        max_mag = max(np.amax(audio), -np.amin(audio))
        max_mag = (
            (2**31) + 1
            if max_mag > (2**15)
            else ((2**15) + 1 if max_mag > 1.01 else 1.0)
        )

    # Normalize the audio
    audio = torch.FloatTensor(audio.astype(np.float32)) / max_mag

    if (torch.isnan(audio) | torch.isinf(audio)).any():
        return [], sample_rate or fs or 48000

    # Resample the audio to our target samplerate
    if fs is not None and fs != sample_rate:
        audio = torch.from_numpy(
            librosa.core.resample(audio.numpy(), orig_sr=sample_rate, target_sr=fs)
        )
        sample_rate = fs

    return audio, fs


def load_audio_torchaudio(wave_file, fs):
    """Load audio data into torch tensor

    Args:
        wave_file (str): path to wave file
        fs (int): sample rate

    Returns:
        audio (tensor): audio data in tensor
        fs (int): sample rate
    """

    audio, sample_rate = torchaudio.load(wave_file)
    assert audio.shape[-1] > 2
    audio = torch.mean(audio, dim=0).squeeze(0)
    if sample_rate != fs:
        audio = torchaudio.functional.resample(
            audio,
            orig_freq=sample_rate,
            new_freq=fs,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    return audio, fs


def load_audio_soundfile(wave_file, fs):
    audio, sr_orig = sf.read(wave_file, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, keepdims=False)
    if sr_orig != fs:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=fs)
    return torch.tensor(audio), fs
