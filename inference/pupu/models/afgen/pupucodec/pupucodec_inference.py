import torch


def pupucodec_inference(cfg, model, audios, device=None):
    model.eval()

    with torch.no_grad():
        output = model.forward(
            audios.unsqueeze(1), n_quantizers=cfg.inference.n_quantizers
        )
        return output["audio"].squeeze(1).detach().cpu()
