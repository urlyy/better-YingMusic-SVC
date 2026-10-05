import os
import torch
import time
import accelerate
import random
import numpy as np
import shutil
import re

from pathlib import Path
from tqdm import tqdm
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from models.afgen.afgen_dataset import (
    AFGenDataset,
    AFGenCollator,
    AFGenConcatDataset,
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from models.vocoders.gan.generator.pupuvocoder import PupuVocoder
from models.afgen.pupucodec.pupucodec import PupuCodec

from models.vocoders.gan.discriminator.mpd import MultiPeriodDiscriminator
from models.vocoders.gan.discriminator.mssbcqtd import MultiScaleSubbandCQTDiscriminator
from models.vocoders.gan.discriminator.msd import MultiScaleDiscriminator
from models.vocoders.gan.discriminator.mssbstftd import (
    MultiBandDiscriminator as MultiScaleSubbandSTFTDiscriminator,
)

from models.afgen.pupuvocoder.pupuvocoder_inference import pupuvocoder_inference
from models.afgen.pupucodec.pupucodec_inference import pupucodec_inference

from utils.io import save_audio
from safetensors.torch import load_file

supported_generators = {
    "pupuvocoder": PupuVocoder,
    "pupucodec": PupuCodec,
}

forward_funcs = {
    "PupuVocoder": pupuvocoder_inference,
    "PupuCodec": pupucodec_inference,
}

supported_discriminators = {
    "mpd": MultiPeriodDiscriminator,
    "msd": MultiScaleDiscriminator,
    "mssbcqtd": MultiScaleSubbandCQTDiscriminator,
    "mssbstftd": MultiScaleSubbandSTFTDiscriminator,
}


class AFGenInference(object):
    def __init__(self, args=None, cfg=None):
        super().__init__()

        start = time.monotonic_ns()
        self.args = args
        self.cfg = cfg

        # Init accelerator
        self.accelerator = accelerate.Accelerator()
        self.accelerator.wait_for_everyone()

        # Get logger
        with self.accelerator.main_process_first():
            self.logger = get_logger("inference", log_level=args.log_level)

        # Init training status
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New inference process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")

        self.vocoder_dir = args.vocoder_dir
        self.logger.debug(f"Vocoder dir: {args.vocoder_dir}")

        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        # Set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # Setup inference mode
        self.cfg.dataset = self.args.infer_datasets

        # Setup data loader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.test_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # Build model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.generator, self.discriminators = self._build_model()
            end = time.monotonic_ns()
            self.logger.info(f"Building model done in {(end - start) / 1e6:.3f}ms")

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

        # Init with accelerate
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        (
            self.test_dataloader,
            self.generator,
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator_scheduler,
            self.discriminator_scheduler,
        ) = self.accelerator.prepare(
            self.test_dataloader,
            self.generator,
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator_scheduler,
            self.discriminator_scheduler,
        )
        for key, discriminator in self.discriminators.items():
            self.discriminators[key] = self.accelerator.prepare_model(discriminator)
        end = time.monotonic_ns()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.3f}ms")

        with self.accelerator.main_process_first():
            self.logger.info("Resuming from checkpoint...")
            start = time.monotonic_ns()
            ckpt_path = Path(args.vocoder_dir)
            if self._is_valid_pattern(ckpt_path.parts[-1]):
                ckpt_path = self._load_model(None, args.vocoder_dir)
            else:
                ckpt_path = self._load_model(args.vocoder_dir)
            end = time.monotonic_ns()
            self.logger.info(
                f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
            )

        self.generator.eval()

        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.exists(os.path.join(args.output_dir, "pred")):
            shutil.rmtree(os.path.join(args.output_dir, "pred"))
        if os.path.exists(os.path.join(args.output_dir, "gt")):
            shutil.rmtree(os.path.join(args.output_dir, "gt"))
        os.makedirs(os.path.join(args.output_dir, "pred"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "gt"), exist_ok=True)

        self.accelerator.wait_for_everyone()

    def _build_test_dataset(self):
        return AFGenDataset, AFGenCollator

    def _build_optimizer(self):
        optimizer_params_generator = [dict(params=self.generator.parameters())]
        generator_optimizer = AdamW(
            optimizer_params_generator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
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
            eps=1e-9,
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

    def _build_model(self):
        try:
            generator = supported_generators[self.cfg.model.generator](self.cfg)
        except:
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

    def _build_dataloader(self):
        """Build dataloader which merges a series of datasets."""
        Dataset, Collator = self._build_test_dataset()

        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset)
            datasets_list.append(subdataset)
        test_dataset = AFGenConcatDataset(datasets_list)
        test_collate = Collator(self.cfg)
        test_batch_size = min(self.cfg.inference.batch_size, len(test_dataset))
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=test_collate,
            num_workers=1,
            batch_size=test_batch_size,
            shuffle=False,
        )
        self.test_batch_size = test_batch_size
        self.test_dataset = test_dataset
        return test_dataloader

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

    def inference(self):
        """Inference via batches"""
        for i, batch in tqdm(enumerate(self.test_dataloader)):
            audio_pred = forward_funcs[self.cfg.model_type](
                self.cfg,
                self.generator,
                batch["audio"],
                device=next(self.generator.parameters()).device,
            )
            audio_ls = audio_pred.chunk(self.test_batch_size)
            audio_gt_ls = batch["audio"].cpu().chunk(self.test_batch_size)
            length_ls = batch["target_len"].cpu().chunk(self.test_batch_size)
            j = 0
            for it, it_gt, l in zip(audio_ls, audio_gt_ls, length_ls):
                l = l.item()
                it = it.squeeze(0).squeeze(0)[:l]
                it_gt = it_gt.squeeze(0)[:l]
                uid = (
                    self.test_dataset.metadata[i * self.test_batch_size + j]
                    .split("/")[-1]
                    .split(".")[0]
                )
                save_audio(
                    os.path.join(self.args.output_dir, "pred", "{}.wav").format(uid),
                    it,
                    self.cfg.preprocess.sample_rate,
                )
                save_audio(
                    os.path.join(self.args.output_dir, "gt", "{}.wav").format(uid),
                    it_gt,
                    self.cfg.preprocess.sample_rate,
                )
                j += 1

    def _set_random_seed(self, seed):
        """Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _is_valid_pattern(self, directory_name):
        directory_name = str(directory_name)
        pattern = r"^epoch-\d{4}_step-\d{7}_loss-\d{1}\.\d{6}"
        return re.match(pattern, directory_name) is not None
