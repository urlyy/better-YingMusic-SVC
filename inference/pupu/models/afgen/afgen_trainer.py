import os
import random
import re

import accelerate
import json5
import numpy as np
import torch
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader

from models.afgen.afgen_dataset import AFGenConcatDataset
from models.afgen.afgen_sampler import build_samplers
from accelerate import DistributedDataParallelKwargs


class AFGenTrainer:
    def __init__(self):
        super().__init__()

    def _init_accelerator(self):
        """Initialize the accelerator components."""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.exp_dir = os.path.join(
            os.path.join(os.environ.get("WORK_DIR"), self.cfg.log_dir),
            self.args.exp_name,
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir, logging_dir=os.path.join(self.exp_dir, "log")
        )
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs],
        )

        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_dataset(self):
        pass

    def _build_criterion(self):
        pass

    def _build_model(self):
        pass

    def _build_dataloader(self):
        """Build dataloader which merges a series of datasets."""
        # Build dataset instance for each dataset and combine them by ConcatDataset
        Dataset, Collator = self._build_dataset()

        # Build train set
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset)
            datasets_list.append(subdataset)
        train_dataset = AFGenConcatDataset(datasets_list)
        train_collate = Collator(self.cfg)
        _, batch_sampler = build_samplers(train_dataset, self.cfg, self.logger, "train")
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.train.dataloader.num_worker,
            pin_memory=self.cfg.train.dataloader.pin_memory,
            persistent_workers=self.cfg.train.dataloader.persistent_workers,
            prefetch_factor=self.cfg.train.dataloader.prefetch_factor,
        )

        return train_loader

    def _build_optimizer(self):
        pass

    def _build_scheduler(self):
        pass

    def _load_model(self, checkpoint_dir, checkpoint_path=None, resume_type="resume"):
        pass

    def train_loop(self):
        pass

    def _train_epoch(self):
        pass

    def _valid_epoch(self):
        pass

    def _train_step(self):
        pass

    def _valid_step(self):
        pass

    def _inference(self):
        pass

    def _set_random_seed(self, seed):
        """Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )

    def _count_parameters(self):
        pass

    def _dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    def _is_valid_pattern(self, directory_name):
        directory_name = str(directory_name)
        pattern = r"^epoch-\d{4}_step-\d{7}_loss-\d{1}\.\d{6}"
        return re.match(pattern, directory_name) is not None
