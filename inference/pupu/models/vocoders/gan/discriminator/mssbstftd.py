import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram

from typing import List, Tuple


class DiscriminatorB(nn.Module):
    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0),
        ),
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length,
            hop_length=int(window_length * hop_factor),
            win_length=window_length,
            power=None,
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))
                ),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        self.conv_post = weight_norm(
            nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1))
        )

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 2, 1)  # [B, F, T, C] -> [B, C, T, F]
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        x = []

        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultiBandDiscriminator(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.fft_sizes = [2048, 1024, 512]
        self.discriminators = nn.ModuleList(
            [DiscriminatorB(window_length=w) for w in self.fft_sizes]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
