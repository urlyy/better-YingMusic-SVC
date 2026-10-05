import torch

from utils.mel import extract_mel_features


def pupuvocoder_inference(cfg, model, audios, device=None):
    model.eval()

    with torch.no_grad():
        mels = extract_mel_features(audios, cfg.preprocess)
        mels = mels.unsqueeze(0).to(device)
        output = model.forward(mels)
        return output.squeeze(1).detach().cpu()
