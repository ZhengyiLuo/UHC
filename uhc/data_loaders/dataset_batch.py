'''
File: /dataset_batch.py
Created Date: Wednesday February 16th 2022
Author: Zhengyi Luo
Comment:
-----
Last Modified: Wednesday February 16th 2022 10:29:09 am
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2022 Carnegie Mellon University, KLab
-----
'''

import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from PIL import Image
import os.path
import torch
import numpy as np
import torch.utils.data as data
import glob
import pickle as pk
import joblib
from collections import defaultdict
from tqdm import tqdm
import ipdb
from multiprocessing import Pool

from uhc.utils.math_utils import (
    de_heading,
    transform_vec,
    quaternion_multiply,
    quaternion_inverse,
    rotation_from_quaternion,
    ewma,
)
import random
from uhc.utils.torch_ext import isNpArray, dict_to_torch, to_numpy


class DatasetBatch(data.Dataset):
    def __init__(self, cfg, data_files, seed=0, multiproess=True):
        np.random.seed(seed)
        print("******* Reading Motion Class Data, Batch Instance! ***********")
        print(data_files)
        self.cfg = cfg
        self.t_min = cfg.data_specs.get("t_min", 15)
        self.t_max = cfg.data_specs.get("t_max", -1)
        self.fr_num = cfg.data_specs.get("fr_num", 90)

        self.data = defaultdict(dict)
        self.data_raw = dict()

        self.multiproess = multiproess

        self.name = " ".join(
            [k.split("/")[-1].split(".")[0] for k in data_files])

        for f in data_files:
            processed_data, raw_data = self.preprocess_data(f)
            processed_data, raw_data = self.post_process_data(
                processed_data, raw_data)
            [self.data[k].update(v) for k, v in processed_data.items()]
            self.data_raw.update(raw_data)

        self.data_keys = list(self.data["pose_aa"].keys())
        self.sample_keys = list(self.data["pose_aa"].keys())

        self.traj_dim = self.data["pose_aa"][self.data_keys[0]].shape[1]
        self.freq_keys = []
        for k, traj in self.data["pose_aa"].items():
            self.freq_keys += [
                k for _ in range(
                    np.ceil(traj.shape[0] / self.fr_num).astype(int))
            ]
        self.freq_keys = np.array(self.freq_keys)
        print("Dataset Root: ", data_files)
        print("Fr_num: ", self.fr_num)
        print("******* Finished AMASS Class Data ***********")

    def post_process_data(self, processed_data, raw_data):
        return processed_data, raw_data

    def preprocess_data(self, data_file):
        data_raw = joblib.load(data_file)
        data_processed = defaultdict(dict)
        all_data = list(data_raw.items())
        if self.multiproess:
            num_jobs = 20
            jobs = all_data
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
            job_args = [(jobs[i], ) for i in range(len(jobs))]
            print(f"Reading data with {len(job_args)} threads")
            try:
                pool = Pool(num_jobs)  # multi-processing
                job_res = pool.starmap(self.process_data_list, job_args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                print(e)
            [[data_processed[k].update(v) for k, v in j.items()]
             for j in job_res]
        else:
            print(f"Reading data with 1 thread")
            data_processed = self.process_data_list(data_list=all_data)

        return data_processed, data_raw

    def process_data_list(self, data_list):
        data_processed = defaultdict(dict)
        # pbar = tqdm(all_data)
        for take, curr_data in data_list:
            pose_aa = curr_data["pose_aa"]
            seq_len = pose_aa.shape[0]
            if seq_len <= self.fr_num:
                print(take, f" too short length: {seq_len} < {self.fr_num}")
                continue

            data_processed["pose_aa"][take] = to_numpy(curr_data["pose_aa"])
            data_processed["pose_6d"][take] = to_numpy(curr_data["pose_6d"])
            data_processed["trans"][take] = to_numpy(curr_data["trans"])

        return data_processed

    def __getitem__(self, index):
        # sample random sequence from data
        take_key = self.sample_keys[index]
        sample = self.get_sample_from_key(take_key,
                                          fr_start=-1,
                                          fr_num=self.fr_num)
        return sample

    def get_seq_len_by_key(self, key):
        return self.data["pose_aa"][key].shape[0]

    def get_seq_len_by_idx(self, idx):
        return self.data["pose_aa"][self.get_seq_key(idx)].shape[0]

    def get_seq_key(self, index):
        return self.data_keys[index]

    def sample_seq(
        self,
        full_sample=False,
        freq_dict=None,
        sampling_temp=0.2,
        sampling_freq=0.5,
        precision_mode=False,
        return_batch=True,
        fr_num=-1,
        fr_start=-1,
    ):

        if freq_dict is None or len(freq_dict.keys()) != len(self.data_keys):
            self.curr_key = curr_key = random.choice(self.sample_keys)
        else:
            init_probs = np.exp(-np.array([
                ewma(np.array(freq_dict[k])[:, 0] == 1)
                if len(freq_dict[k]) > 0 else 0 for k in freq_dict.keys()
            ]) / sampling_temp)
            init_probs = init_probs / init_probs.sum()
            self.curr_key = curr_key = (np.random.choice(
                self.data_keys, p=init_probs) if np.random.binomial(
                    1, sampling_freq) else np.random.choice(self.data_keys))
        curr_pose_aa = self.data["pose_aa"][self.curr_key]
        seq_len = curr_pose_aa.shape[0]

        return self.get_sample_from_key(self.curr_key,
                                        full_sample=full_sample,
                                        precision_mode=precision_mode,
                                        fr_num=fr_num,
                                        freq_dict=freq_dict,
                                        sampling_freq=sampling_freq,
                                        return_batch=return_batch,
                                        fr_start=fr_start)

    def get_sample_from_key(self,
                            take_key,
                            full_sample=False,
                            full_fr_num=False,
                            freq_dict=None,
                            fr_start=-1,
                            fr_num=-1,
                            precision_mode=False,
                            sampling_freq=0.75,
                            return_batch=False,
                            exclude_keys=[]):
        """_summary_

        Args:
            take_key (_type_): _description_
            full_sample (bool, optional): _description_. Defaults to False.
            freq_dict (_type_, optional): _description_. Defaults to None.
            fr_start (int, optional): _description_. Defaults to -1.
            fr_num (int, optional): _description_. Defaults to -1. if != -1, it will be used to constraint the length of the data
            precision_mode (bool, optional): _description_. Defaults to False.
            sampling_freq (float, optional): _description_. Defaults to 0.75.

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        self.curr_take_ind = self.data_keys.index(take_key)
        self.curr_key = take_key
        if not take_key in self.data["pose_aa"]:
            raise Exception("Key not found")

        curr_qpos = self.data["pose_aa"][take_key]
        seq_len = curr_qpos.shape[0]

        if full_sample:
            self.fr_start = fr_start = 0
            # self.fr_start = fr_start = 1700
            self.fr_end = fr_end = self.data["pose_aa"][self.curr_key].shape[0]
        else:
            if not freq_dict is None and precision_mode:
                perfs = np.array(freq_dict[take_key])
                if (len(perfs) > 0 and len(perfs[perfs[:, 0] != 1][:, 1]) > 0
                        and np.random.binomial(1, sampling_freq)):
                    perfs = perfs[perfs[:, 0] != 1][:, 1]
                    chosen_idx = np.random.choice(perfs)
                    self.fr_start = fr_start = np.random.randint(
                        max(chosen_idx - 20 - self.t_min, 0),
                        min(chosen_idx + 20, seq_len - self.t_min),
                    )
                else:
                    self.fr_start = fr_start = np.random.randint(
                        0, seq_len - self.t_min)
            elif fr_start == -1:
                self.fr_start = fr_start = np.random.randint(
                    0, seq_len - (self.t_min if not full_fr_num else fr_num))
            else:
                self.fr_start = fr_start

            if fr_num == -1:
                self.fr_end = fr_end = (fr_start + self.t_max if
                                        (fr_start + self.t_max < seq_len
                                         and self.t_max != -1) else seq_len)
            else:
                self.fr_end = fr_end = fr_start + fr_num

        sample = {}

        for key in self.data.keys():
            if not key in exclude_keys:
                sample[key] = self.data[key][self.curr_key][fr_start:fr_end]

        sample["seq_name"] = self.curr_key

        if return_batch:
            data_sample = dict_to_torch(sample, add_dim = True)
            return data_sample
        else:
            return sample

    def get_key_by_ind(self, ind):
        return self.data_keys[ind]

    def get_seq_by_ind(self, ind, full_sample=False):
        take_key = self.data_keys[ind]
        data_dict = self.get_sample_from_key(take_key, full_sample=full_sample)
        return {k: torch.from_numpy(v)[None, ] for k, v in data_dict.items()}

    def get_len(self):
        return len(self.data_keys)

    def sampling_loader(self,
                        batch_size=8,
                        num_samples=5000,
                        num_workers=1,
                        fr_num=80):
        self.fr_num = int(fr_num)
        self.sample_keys = np.random.choice(self.freq_keys,
                                            num_samples,
                                            replace=True)
        self.data_len = len(self.sample_keys)  # Change sequence length
        loader = torch.utils.data.DataLoader(self,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
        return loader

    def iter_loader(self, batch_size=8, num_workers=1, fr_num=80):
        # Not really iter...
        self.fr_num = int(fr_num)
        self.data_curr = [
            i for i in self.freq_keys
            if self.data["pose_aa"][i].shape[0] >= fr_num
        ]
        self.sample_keys = self.data_curr
        self.data_len = len(self.sample_keys)  # Change sequence length
        loader = torch.utils.data.DataLoader(self,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
        return loader

    def __len__(self):
        return self.data_len

    def iter_data(self):
        data = {}
        for take_key in self.data_keys:
            self.curr_key = take_key
            seq_len = self.data["pose_aa"][take_key].shape[
                0]  # not using the fr_num at all
            data_return = {}

            for k in self.data.keys():
                data_return[k] = self.data[k][take_key]

            data[take_key] = {
                k: torch.from_numpy(v)[None, ]
                for k, v in data_return.items()
            }
            data[take_key]['seq_name'] = take_key
        return data

    def get_data(self):
        return self.data

    def get_sample_len_from_key(self, take_key):
        return self.data["pose_aa"][take_key].shape[0]


if __name__ == "__main__":
    np.random.seed(0)
    from uhc.utils.config_utils.uhm_config import Config

    cfg = Config(cfg_id="uhm_init", create_dirs=False)

    dataset = DatasetBatch(cfg)
    for i in range(10):
        generator = dataset.sampling_loader(num_samples=5000,
                                            batch_size=1,
                                            num_workers=1)
        for data in generator:
            import pdb

            pdb.set_trace()
        print("-------")