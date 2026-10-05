import os
import time
import torch
from tqdm import tqdm
from torch import nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from librosa.filters import mel as librosa_mel_fn

from accelerate.logging import get_logger
from pathlib import Path

from models.afgen.afgen_trainer import AFGenTrainer
from models.afgen.pupucodec.pupucodec_dataset import (
    PupuCodecDataset,
    PupuCodecCollator,
)

from models.afgen.pupucodec.pupucodec import PupuCodec

from models.vocoders.gan.discriminator.mpd import MultiPeriodDiscriminator
from models.vocoders.gan.discriminator.mssbcqtd import MultiScaleSubbandCQTDiscriminator
from models.vocoders.gan.discriminator.msd import MultiScaleDiscriminator
from models.vocoders.gan.discriminator.mssbstftd import (
    MultiBandDiscriminator as MultiScaleSubbandSTFTDiscriminator,
)

from scipy import signal
import typing
from typing import List
from collections import namedtuple
import math
import functools
from safetensors.torch import load_file

supported_generators = {
    "pupucodec": PupuCodec,
}

supported_discriminators = {
    "mpd": MultiPeriodDiscriminator,
    "msd": MultiScaleDiscriminator,
    "mssbcqtd": MultiScaleSubbandCQTDiscriminator,
    "mssbstftd": MultiScaleSubbandSTFTDiscriminator,
}


class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(
                x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(
                y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))

            loss = loss + self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss = loss + self.mag_weight * self.loss_fn(x_logmels, y_logmels)

        return loss * 15


class PupuCodecTrainer(AFGenTrainer):
    def __init__(self, args, cfg):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # Init accelerator
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            self.logger = get_logger(args.exp_name, log_level=args.log_level)

        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New training process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.debug(f"Using {args.log_level.upper()} logging level.")
        self.logger.info(f"Experiment name: {args.exp_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Init training status
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0

        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        self.logger.info(
            "Max epoch: {}".format(
                self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
            )
        )

        # Check potential erorrs
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.run_eval = self.cfg.train.run_eval

        # Set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # Build dataloader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # Build model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.generator, self.discriminators = self._build_model()
            end = time.monotonic_ns()
            self.logger.debug(self.generator)
            for _, discriminator in self.discriminators.items():
                self.logger.debug(discriminator)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
            self.logger.info(f"Model parameters: {self._count_parameters()/1e6:.2f}M")

        # Build optimizers and schedulers
        with self.accelerator.main_process_first():
            self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            (
                self.generator_optimizer,
                self.discriminator_optimizer,
            ) = self._build_optimizer()
            (
                self.generator_scheduler,
                self.discriminator_scheduler,
            ) = self._build_scheduler()
            end = time.monotonic_ns()
            self.logger.info(
                f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
            )

        # Accelerator preparing
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        (
            self.train_dataloader,
            self.generator,
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator_scheduler,
            self.discriminator_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.generator,
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator_scheduler,
            self.discriminator_scheduler,
        )
        for key, discriminator in self.discriminators.items():
            self.discriminators[key] = self.accelerator.prepare_model(discriminator)
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # Build criterions
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterions = self._build_criterion()
            end = time.monotonic_ns()
            self.logger.info(f"Building criterion done in {(end - start) / 1e6:.2f}ms")

        # Resume checkpoints
        with self.accelerator.main_process_first():
            if args.resume_type:
                self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = Path(args.checkpoint)
                if self._is_valid_pattern(ckpt_path.parts[-1]):
                    ckpt_path = self._load_model(
                        None, args.checkpoint, args.resume_type
                    )
                else:
                    ckpt_path = self._load_model(
                        args.checkpoint, resume_type=args.resume_type
                    )
                end = time.monotonic_ns()
                self.logger.info(
                    f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Save config
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        try:
            torch.distributed.barrier()
        except:
            pass

    def _build_dataset(self):
        return PupuCodecDataset, PupuCodecCollator

    def _build_criterion(self):
        class feature_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(feature_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, fmap_r, fmap_g):
                loss = 0

                for dr, dg in zip(fmap_r, fmap_g):
                    for rl, gl in zip(dr, dg):
                        loss = loss + self.l1Loss(rl, gl)
                loss = loss * 2

                return loss

        class discriminator_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(discriminator_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, disc_real_outputs, disc_generated_outputs):
                loss = 0
                r_losses = []
                g_losses = []

                for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                    r_loss = torch.mean((1 - dr) ** 2)
                    g_loss = torch.mean(dg**2)
                    loss = loss + r_loss + g_loss
                    r_losses.append(r_loss.item())
                    g_losses.append(g_loss.item())

                return loss, r_losses, g_losses

        class generator_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(generator_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, disc_outputs):
                loss = 0
                gen_losses = []

                for dg in disc_outputs:
                    l = torch.mean((1 - dg) ** 2)
                    gen_losses.append(l)
                    loss = loss + l

                return loss, gen_losses

        criterions = dict()
        for key in self.cfg.train.criterions:
            if key == "feature":
                criterions["feature"] = feature_criterion(self.cfg)
            elif key == "discriminator":
                criterions["discriminator"] = discriminator_criterion(self.cfg)
            elif key == "generator":
                criterions["generator"] = generator_criterion(self.cfg)
            elif key == "multimel":
                criterions["multimel"] = MultiScaleMelSpectrogramLoss(
                    self.cfg.preprocess.sample_rate
                )
            elif key == "commitment":
                criterions["commitment"] = None
            elif key == "codebook":
                criterions["codebook"] = None
            else:
                raise NotImplementedError

        return criterions

    def _build_model(self):
        generator = supported_generators[self.cfg.model.generator](
            encoder_dim=self.cfg.model.pupucodec.encoder_dim,
            encoder_rates=self.cfg.model.pupucodec.encoder_rates,
            decoder_dim=self.cfg.model.pupucodec.decoder_dim,
            decoder_rates=self.cfg.model.pupucodec.decoder_rates,
            n_codebooks=self.cfg.model.pupucodec.n_codebooks,
            codebook_size=self.cfg.model.pupucodec.codebook_size,
            codebook_dim=self.cfg.model.pupucodec.codebook_dim,
            quantizer_dropout=self.cfg.model.pupucodec.quantizer_dropout,
        )
        discriminators = dict()
        for key in self.cfg.model.discriminators:
            discriminators[key] = supported_discriminators[key](self.cfg)

        return generator, discriminators

    def _build_optimizer(self):
        optimizer_params_generator = [dict(params=self.generator.parameters())]
        generator_optimizer = AdamW(
            optimizer_params_generator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
            fused=True,
        )

        optimizer_params_discriminator = []
        for discriminator in self.discriminators.keys():
            optimizer_params_discriminator.append(
                dict(params=self.discriminators[discriminator].parameters())
            )
        discriminator_optimizer = AdamW(
            optimizer_params_discriminator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
            fused=True,
        )

        return generator_optimizer, discriminator_optimizer

    def _build_scheduler(self):
        generator_scheduler = ExponentialLR(
            self.generator_optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch - 1,
        )

        discriminator_scheduler = ExponentialLR(
            self.discriminator_optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch - 1,
        )

        return generator_scheduler, discriminator_scheduler

    def train_loop(self):
        """Training process"""
        self.accelerator.wait_for_everyone()

        # Dump config
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)

        self.generator.train()
        for key in self.discriminators.keys():
            self.discriminators[key].train()
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        # Sync and start training
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            # Train
            train_total_loss, train_losses = self._train_epoch()
            for key, loss in train_losses.items():
                self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                self.accelerator.log(
                    {"Epoch/Train {} Loss".format(key): loss},
                    step=self.epoch,
                )
            self.accelerator.log(
                {
                    "Epoch/Train Total Loss": train_total_loss,
                },
                step=self.epoch,
            )

            # Update scheduler
            self.accelerator.wait_for_everyone()
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            # Check save checkpoint interval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        run_eval |= self.run_eval[i]

            # Save checkpoints
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
                pkg = dict(
                    generator=self.accelerator.get_state_dict(self.generator),
                    discriminators={
                        k: self.accelerator.get_state_dict(v)
                        for k, v in self.discriminators.items()
                    },
                    generator_optimizer=self.generator_optimizer.state_dict(),
                    discriminator_optimizer=self.discriminator_optimizer.state_dict(),
                    generator_scheduler=self.generator_scheduler.state_dict(),
                    discriminator_scheduler=self.discriminator_scheduler.state_dict(),
                )
                torch.save(pkg, path)

            self.accelerator.wait_for_everyone()

            self.epoch = self.epoch + 1

    def _train_epoch(self):
        """Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.generator.train()
        for key, _ in self.discriminators.items():
            self.discriminators[key].train()

        epoch_losses: dict = {}
        epoch_total_loss: int = 0

        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Get losses
            total_loss, losses = self._train_step(batch)
            self.batch_count = self.batch_count + 1

            # Log info
            self.accelerator.log(
                {
                    "Step/Generator Learning Rate": self.generator_optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ],
                    "Step/Discriminator Learning Rate": self.discriminator_optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ],
                },
                step=self.step,
            )
            for key, _ in losses.items():
                self.accelerator.log(
                    {
                        "Step/Train {} Loss".format(key): losses[key],
                    },
                    step=self.step,
                )

            if not epoch_losses:
                epoch_losses = losses
            else:
                for key, value in losses.items():
                    if not key in epoch_losses.keys():
                        epoch_losses[key] = 0
                    epoch_losses[key] = epoch_losses[key] + value
            epoch_total_loss = epoch_total_loss + total_loss
            self.step = self.step + 1

        # Get and log total losses
        self.accelerator.wait_for_everyone()
        epoch_total_loss = (
            epoch_total_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )
        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )
        return epoch_total_loss, epoch_losses

    def _train_step(self, data):
        """Training forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_train_epoch`` for usage.
        """
        # Init losses
        train_losses = {}
        total_loss = 0

        generator_losses = {}
        generator_total_loss = 0
        discriminator_losses = {}
        discriminator_total_loss = 0

        # Get feature
        audio_gt = data["audio"].unsqueeze(1)
        output = self.generator.forward(audio_gt)
        audio_pred, generator_losses["commitment"], generator_losses["codebook"] = (
            output["audio"],
            output["vq/commitment_loss"],
            output["vq/codebook_loss"],
        )

        # Calculate and BP Discriminator losses
        self.discriminator_optimizer.zero_grad()
        for key, _ in self.discriminators.items():
            y_r, y_g, _, _ = self.discriminators[key].forward(
                audio_gt, audio_pred.detach()
            )
            (
                discriminator_losses["{}_discriminator".format(key)],
                _,
                _,
            ) = self.criterions["discriminator"](y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_discriminator".format(key)
            ]
        self.accelerator.backward(discriminator_total_loss)
        self.discriminator_optimizer.step()

        # Calculate and BP Generator losses
        self.generator_optimizer.zero_grad()
        for key, _ in self.discriminators.items():
            y_r, y_g, f_r, f_g = self.discriminators[key].forward(audio_gt, audio_pred)
            generator_losses["{}_feature".format(key)] = self.criterions["feature"](
                f_r, f_g
            )
            generator_losses["{}_generator".format(key)], _ = self.criterions[
                "generator"
            ](y_g)
            generator_total_loss += generator_losses["{}_feature".format(key)]
            generator_total_loss += generator_losses["{}_generator".format(key)]

        if "multimel" in self.criterions.keys():
            generator_losses["multimel"] = self.criterions["multimel"](
                audio_gt, audio_pred
            )
            generator_total_loss += generator_losses["multimel"]

        if "commitment" in self.criterions.keys():
            generator_total_loss += generator_losses["commitment"].squeeze() * 0.25

        if "codebook" in self.criterions.keys():
            generator_total_loss += generator_losses["codebook"].squeeze()

        self.accelerator.backward(generator_total_loss)
        self.generator_optimizer.step()

        # Get the total losses
        total_loss = discriminator_total_loss + generator_total_loss
        train_losses.update(discriminator_losses)
        train_losses.update(generator_losses)

        for key, _ in train_losses.items():
            train_losses[key] = train_losses[key].item()

        return total_loss.item(), train_losses

    def _load_model(self, checkpoint_dir, checkpoint_path=None, resume_type="resume"):
        """Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
            ls.sort(key=lambda x: int(x.split("_")[-3].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
        if resume_type == "resume":
            try:
                self.accelerator.load_state(checkpoint_path)
            except:
                pkg = torch.load(checkpoint_path, map_location="cpu")
                self.accelerator.unwrap_model(self.generator).load_state_dict(
                    pkg["generator"]
                )
                for key, _ in self.discriminators.items():
                    self.accelerator.unwrap_model(
                        self.discriminators[key]
                    ).load_state_dict(pkg["discriminators"][key])
                self.generator_optimizer.load_state_dict(pkg["generator_optimizer"])
                self.generator_scheduler.load_state_dict(pkg["generator_scheduler"])
                self.discriminator_optimizer.load_state_dict(
                    pkg["discriminator_optimizer"]
                )
                self.discriminator_scheduler.load_state_dict(
                    pkg["discriminator_scheduler"]
                )
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1]) + 1
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1]) + 1
        elif resume_type == "finetune":
            try:
                state_dict = load_file(
                    os.path.join(checkpoint_path, f"model.safetensors")
                )
                state_dict = {
                    f"module.{key}": value for key, value in state_dict.items()
                }
                self.generator.load_state_dict(state_dict)
                for index, (key, _) in enumerate(self.discriminators.items()):
                    state_dict = load_file(
                        os.path.join(checkpoint_path, f"model_{index + 1}.safetensors")
                    )
                    state_dict = {
                        f"module.{key}": value for key, value in state_dict.items()
                    }
                    self.discriminators[key].load_state_dict(state_dict)
            except:
                pkg = torch.load(checkpoint_path, map_location="cpu")
                self.accelerator.unwrap_model(self.generator).load_state_dict(
                    pkg["generator"]
                )
                for key, _ in self.discriminators.items():
                    self.accelerator.unwrap_model(
                        self.discriminators[key]
                    ).load_state_dict(pkg["discriminators"][key])
        else:
            raise ValueError("Unsupported resume type: {}".format(resume_type))
        return checkpoint_path

    def _count_parameters(self):
        result = sum(p.numel() for p in self.generator.parameters())
        for _, discriminator in self.discriminators.items():
            result = result + sum(p.numel() for p in discriminator.parameters())
        return result
