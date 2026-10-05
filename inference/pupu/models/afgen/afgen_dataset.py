from typing import Iterable
import torch
import numpy as np
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset
from utils.audio import load_audio_torch
import os
import os.path as osp
import json


class AFGenDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset):
        assert isinstance(dataset, str)
        self.work_dir = os.environ.get("WORK_DIR")
        self.dataset_json = json.load(
            open(f"{self.work_dir}/dataset/{dataset}.json", "r")
        )
        self.dataset_list = self.dataset_json["datasets"]
        self.filelist_path = self.dataset_json["filelist_path"]

        self.dataset = dataset
        self.metadata = self.get_metadata()
        self.length = len(self.metadata)

        self.cfg = cfg

    def __getitem__(self, index):
        single_feature = dict()

        ceph_dir = self.metadata[index]
        audio_torch, _ = load_audio_torch(ceph_dir, self.cfg.preprocess.sample_rate)
        audio = audio_torch.cpu().numpy()

        single_feature["target_len"] = audio.shape[-1]
        if audio.shape[-1] % self.cfg.preprocess.hop_size != 0:
            audio = np.pad(
                audio,
                (
                    (
                        0,
                        self.cfg.preprocess.hop_size
                        - audio.shape[-1] % self.cfg.preprocess.hop_size,
                    )
                ),
                mode="constant",
            )
        single_feature["audio"] = audio

        return single_feature

    def get_metadata(self):
        file_list = []

        for dataset in self.dataset_list:
            filelist_json = osp.join(
                self.work_dir, self.filelist_path, dataset["name"] + ".json"
            )
            wav_data_list = json.load(open(filelist_json, "r"))
            for wav_data in wav_data_list:
                if "data_path" in wav_data:
                    wav = wav_data["data_path"]
                elif "path" in wav_data:
                    wav = wav_data["path"]
                elif "id" in wav_data:
                    wav = wav_data["id"] + ".wav"
                if dataset["path"] != "":
                    wav_path = osp.join(dataset["path"], wav)
                else:
                    wav_path = wav

                file_list.append(wav_path)

        return file_list

    def get_dataset_name(self):
        return self.dataset

    def __len__(self):
        return self.length


class AFGenConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__(datasets)

        self.cfg = self.datasets[0].cfg

        self.metadata = []

        for dataset in self.datasets:
            self.metadata += dataset.metadata


class AFGenCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
