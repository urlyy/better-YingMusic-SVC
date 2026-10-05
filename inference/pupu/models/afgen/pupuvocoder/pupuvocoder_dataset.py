import torch
import random
import json
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from models.afgen.afgen_dataset import AFGenDataset
from utils.audio import load_audio_torchaudio
import torch
import io
import os.path as osp

from utils.mel import extract_mel_features


class PupuVocoderDataset(AFGenDataset):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset)

    def get_batch(self, index):
        single_feature = dict()

        ceph_dir = self.metadata[index]
        audio_torch, _ = load_audio_torchaudio(
            ceph_dir, self.cfg.preprocess.sample_rate
        )
        audio = audio_torch.numpy()

        if (
            audio.shape[-1]
            <= self.cfg.preprocess.cut_mel_frame * self.cfg.preprocess.hop_size
        ):
            audio = np.pad(
                audio,
                (
                    (
                        0,
                        self.cfg.preprocess.cut_mel_frame * self.cfg.preprocess.hop_size
                        - audio.shape[-1],
                    )
                ),
                mode="constant",
            )
        else:
            mel_length = audio.shape[-1] // self.cfg.preprocess.hop_size
            start = random.randint(0, mel_length - self.cfg.preprocess.cut_mel_frame)
            end = start + self.cfg.preprocess.cut_mel_frame
            audio = audio[
                start
                * self.cfg.preprocess.hop_size : end
                * self.cfg.preprocess.hop_size,
            ]
        single_feature["audio"] = audio
        single_feature["mel"] = extract_mel_features(
            torch.from_numpy(audio).unsqueeze(0), self.cfg.preprocess
        )
        single_feature["mel"] = single_feature["mel"].squeeze().numpy()
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

        random.shuffle(file_list)

        return file_list

    def __getitem__(self, index):
        while True:
            try:
                batch = self.get_batch(index)
                break
            except Exception as e:
                print(self.metadata[index], flush=True)
                print(e, flush=True)
                index = random.randint(0, self.length - 1)
        return batch


class PupuVocoderCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key in ["target_len", "start", "end"]:
                continue
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
