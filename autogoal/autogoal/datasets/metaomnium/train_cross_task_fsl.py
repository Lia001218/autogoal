# Our code builds on https://github.com/ihsaan-ullah/meta-album

# -----------------------
# Imports
# -----------------------

import os
import sys

# In order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import datetime
import json

import pickle
import random
import time
from copy import deepcopy

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import tqdm
from typing import Any
from .data_loader_cross_problem_fsl import (
    DataLoaderCrossProblem,
    create_datasets,
    create_datasets_task_type,
    process_labels,
    get_k_keypoints,
)


class CrossTaskFewShotLearningExperiment:
    def __init__(self, args, datasets_folder: str, seed: int = 42, root_dir: str = 'None'):
        self.args = args
        self.seed = seed
        # Define paths
        self.curr_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if root_dir != "None":
            self.main_dir = self.args.root_dir
        else:
            self.main_dir = self.curr_dir

        self.res_dir = os.path.join(self.main_dir, "results")
        self.data_dir = os.path.join(self.main_dir, "data")
        self.logs_path = os.path.join(self.main_dir, "logs")
        trains_datasets = ''
        for dataset in os.listdir(self.data_dir):
            if dataset.startswith('train'):
                trains_datasets += dataset + ','
        self.train_datasets = trains_datasets
        # Initialization step
        
        self.set_seed()
        self.clprint = lambda text: lprint(text, self.logs_path)
        self.configure()

    
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def configure(self):
        (
            train_datasets,
            train_dataset_task_type_dict,
            weights,
        ) = create_datasets_task_type(
            self.train_datasets.split(","), self.data_dir
        )
        if "segmentation" in train_datasets:
            self.args.segm_classes = (
                len(train_datasets["segmentation"].idx_per_label) + 1
            )  # one class for background
        train_datasets = list(train_datasets.values())
        
        self.train_loader = DataLoaderCrossProblem(
            train_datasets,
            self.args.train_iters,
            self.args.train_episodes_config,
            weights=weights,
        )

        self.dataset_task_type_dict = {}
        for dataset in train_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = train_dataset_task_type_dict[dataset]

        # Print the configuration for confirmation
        self.clprint("\n\n### ------------------------------------------ ###")
        self.clprint(f"Model: {self.args.model}")
        self.clprint(f"Training Datasets: {self.args.train_datasets}")
        self.clprint(f"In-Domain Validation Datasets: {self.args.val_id_datasets}")
        self.clprint(f"Out-Domain Validation Datasets: {self.args.val_od_datasets}")
        self.clprint(f"In-Domain Testing Datasets: {self.args.test_id_datasets}")
        self.clprint(f"Out-Domain Testing Datasets: {self.args.test_od_datasets}")
        self.clprint(f"Random Seed: {self.args.seed}")
        self.clprint("### ------------------------------------------ ###\n")

    def overwrite_conf(self, conf, arg_str):
        # If value provided in arguments, overwrite the config with it
        value = getattr(self.args, arg_str)
        if value is not None:
            conf[arg_str] = value
        else:
            if arg_str not in conf:
                conf[arg_str] = None
            else:
                setattr(self.args, arg_str, conf[arg_str])

    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

    def run(self):
        # seeds = [random.randint(0, 100000) for _ in range(self.args.runs)]
        # print(f"Run seeds: {seeds}")

        # for run in range(self.args.runs):
        #     self.clprint("\n\n" + "-" * 40)
        #     self.clprint(f"[*] Starting run {run} with seed {seeds[run]}")

        torch.manual_seed(self.seed)
        train_generator = iter(self.train_loader.generator(self.seed))
        # with tqdm.tqdm(total=self.args.train_iters) as pbar_epochs:
        for i, task in enumerate(train_generator):
            ttime = time.time()
            n_way = task.n_way
            k_shot = task.k_shot
            query_size = task.query_size
            data = task.data
            labels = task.labels
            support_size = n_way * k_shot
            
            # Process the labels according to the task type
            if task.task_type == "segmentation":
                labels = task.segmentations.squeeze(dim=1)
            elif task.task_type.startswith("regression"):
                if task.task_type in [
                    "regression_pose_animals",
                    "regression_pose_animals_syn",
                    "regression_mpii",
                ]:
                    labels = get_k_keypoints(
                        n_way * 5, task.regressions, task.task_type
                    )
                else:
                    labels = task.regressions
            else:
                labels = process_labels(
                    n_way * (k_shot + query_size), n_way
                )
            train_x, train_y, test_x, test_y = (
                data[:support_size],
                labels[:support_size],
                data[support_size:],
                labels[support_size:],
            )
            yield train_x, train_y



def set_random_seeds(random_seed: int) -> None:
    if random_seed is not None:
        torch.backends.cudnn.deterministic = False
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)


def lprint(text: str, 
           logs_path: str) -> None:
    print(text)
    with open(logs_path, "a") as f:
        f.write(text + "\n")


def get_device(logs_path: str) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        lprint(f"Using GPU: {torch.cuda.get_device_name(device)}", logs_path)
    else:
        device = torch.device("cpu")
        lprint("Using CPU", logs_path)
    return device


def get_torch_gpu_environment() -> list[str]:
    env_info = list()
    env_info.append(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        env_info.append(f"Cuda version: {torch.version.cuda}")
        env_info.append(f"cuDNN version: {torch.backends.cudnn.version()}")
        env_info.append("Number of available GPUs: "
            + f"{torch.cuda.device_count()}")
        env_info.append("Current GPU name: " +
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        env_info.append("Number of available GPUs: 0")
    
    return env_info


def create_results_dir(res_dir: str,
                       logs_path: str) -> None:
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        lprint(f"[+] Results directory created: {res_dir}", logs_path)
    else:
        lprint(f"[!] Results directory already exists: {res_dir}", logs_path)
        
        
def create_dir(dirname: str) -> None:
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass


def count_trainable_parameters(model: Any) -> int:
    return sum([x.numel() for x in model.parameters() if x.requires_grad])
