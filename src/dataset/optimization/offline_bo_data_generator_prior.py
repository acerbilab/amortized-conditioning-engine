from typing import List, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from src.dataset.optimization.bo_data_generator_prior import (
    BayesianOptimizationDataGeneratorPrior,
)
from itertools import cycle
import hydra
import time
import math
import yaml
import os
import copy
import random

MAX_SAMPLE_RETRIES = 5


class OfflineOptDataGenerator:
    def __init__(
        self,
        num_data,
        num_vcpu,
        num_processes,
        mp_method="spawn",
        data_dir=None,
        verbose=False,
        **data_generator_params,
    ) -> None:
        self.num_data = num_data
        self.num_processes = num_processes
        self.mp_method = mp_method

        self.mp_method = "spawn"
        self.verbose = verbose
        self.num_threads = max(1, math.floor(num_vcpu / num_processes))
        self.data_dir = data_dir

        self.batch_size = data_generator_params["batch_size"]
        self.n_total_points = data_generator_params["n_total_points"]
        self.num_latent = data_generator_params["num_latent"]
        self.max_ctx_points = data_generator_params["max_ctx_points"]
        self.num_bins = data_generator_params["num_bins"]
        self.x_range = data_generator_params["x_range"]
        self.device = data_generator_params["device"]

        data_generator_params["x_range"] = torch.Tensor(
            data_generator_params["x_range"]
        ).tolist()
        os.makedirs(data_dir, exist_ok=True)
        with open(f"{data_dir}dataset.yaml", "w") as file:
            yaml.dump(data_generator_params, file)

    def run(self):
        for i_rank in range(0, self.num_data, self.num_processes):
            self.paralel_run(i_rank)

    def paralel_run(self, i_rank):
        if self.verbose:
            start_time = time.time()
        processes = []

        for rank in range(i_rank, i_rank + self.num_processes):
            seed = torch.initial_seed() + rank
            pr = mp.Process(target=self.worker_process, args=(rank, seed))
            pr.start()
            processes.append(pr)

        for pr in processes:
            pr.join()

        if self.verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"All processes completed and data saved.")
            print(f"Total time taken: {elapsed_time:.2f} seconds")

    def worker_process(self, rank, seed):
        torch.manual_seed(seed)
        # np.random.seed(seed)
        random.seed(seed)
        torch.set_num_threads(self.num_threads)
        if self.verbose:
            print(f"Process {rank} started")
        data = self.get_data()
        if self.verbose:
            print(f"Process {rank} finished")
        file_name = f"{self.data_dir}optim{self.num_latent-1}D_{rank+1}_data.pt"

        torch.save(data, file_name)
        return data

    def _get_data(self):
        """
        This function is a single run for data generator. Note that we set n_ctx_points
        to max_ctx_points so it can be use arbritariry in the ctx tar sampler.
        """
        dataset = BayesianOptimizationDataGeneratorPrior()

        batch_xyd, batch_xyl, latent_bin_weight = dataset.get_data(
            batch_size=self.batch_size,
            n_total_points=self.n_total_points - self.num_latent,
            n_ctx_points=self.max_ctx_points,  # we assume to always have max ctx
            num_bins=self.num_bins,
            x_range=self.x_range,
            device=self.device,
        )
        return batch_xyd, batch_xyl, latent_bin_weight

    def get_data(self):
        attempts = 0
        while attempts < MAX_SAMPLE_RETRIES:
            try:
                attempts += 1
                batch_xyd, batch_xyl, latent_bin_weight = self._get_data()
                check_tensors(batch_xyd, batch_xyl, latent_bin_weight)
            except RuntimeError as e:
                print(f"Attempt {attempts} failed with error: {e}")
                if attempts >= MAX_SAMPLE_RETRIES:
                    print("Max retries reached. Aborting.")
                    raise e

        return batch_xyd, batch_xyl, latent_bin_weight


def check_tensors(*tensors):
    for tensor in tensors:
        if torch.isnan(tensor).sum() > 0:
            raise ValueError(f"Tensor contains NaN values: {tensor}")
        if torch.isinf(tensor).sum() > 0:
            raise ValueError(f"Tensor contains infinite values: {tensor}")


class FilePerBatchDataset(Dataset):
    def __init__(self, data_dir, file_paths, samples_per_file):
        self.data_dir = data_dir
        self.file_paths = file_paths
        self.samples_per_file = samples_per_file
        self.total_samples = len(file_paths) * self.samples_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which file this sample belongs to
        file_idx = idx // self.samples_per_file
        sample_idx_within_file = idx % self.samples_per_file
        # Load the corresponding file
        file_path = os.path.join(self.data_dir, self.file_paths[file_idx])
        data = torch.load(file_path)
        xyd, xyl, bin_weight = data
        # Return the specific sample from the file
        return (
            xyd[sample_idx_within_file],
            xyl[sample_idx_within_file],
            bin_weight[sample_idx_within_file],
        )

    def load_yaml(self, path):
        with open(f"{path}/dataset.yaml", "r") as file:
            loaded_data = yaml.safe_load(file)
        return loaded_data


class FilePerBatchDatasetNoPrior(FilePerBatchDataset):
    def __init__(self, data_dir, file_paths, samples_per_file):
        super().__init__(data_dir, file_paths, samples_per_file)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which file this sample belongs to
        file_idx = idx // self.samples_per_file
        sample_idx_within_file = idx % self.samples_per_file
        # Load the corresponding file
        file_path = os.path.join(self.data_dir, self.file_paths[file_idx])
        data = torch.load(file_path)
        xyd, xyl, _ = data
        # Return the specific sample from the file
        return (
            xyd[sample_idx_within_file],
            xyl[sample_idx_within_file],
        )


class ProblemWrapper:
    def __init__(
        self,
        data_dir,
        data_batch_size,
        train_batch_size,
        eval_batch_size,
        eval_split,
        train_test_split=0.8,
        num_workers=4,
        prior=True,
    ) -> None:
        self.dataset_info = self.load_yaml(data_dir)
        self.data_dir = data_dir
        file_paths = [f for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.data_batch_size = data_batch_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size / eval_split

        num_files = len(file_paths)

        train_idx = int(num_files * train_test_split)

        train_file_paths = file_paths[:train_idx]
        eval_file_paths = file_paths[train_idx:]

        if prior:
            train_dataset = FilePerBatchDataset(
                data_dir, train_file_paths, data_batch_size
            )
            eval_dataset = FilePerBatchDataset(
                data_dir, eval_file_paths, data_batch_size
            )
        else:
            train_dataset = FilePerBatchDatasetNoPrior(
                data_dir, train_file_paths, data_batch_size
            )
            eval_dataset = FilePerBatchDatasetNoPrior(
                data_dir, eval_file_paths, data_batch_size
            )

        # make sure `suffle = False`` for faster data loading (less torch.load calls)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        self.train_dataiter = iter(self.train_dataloader)
        self.eval_dataiter = iter(self.eval_dataloader)

        self.dataset_info = self.load_yaml(data_dir)

    def _get_data(self, data_iter):
        return next(data_iter)

    def get_data(
        self,
        batch_size: int,
        n_total_points: int,
        n_ctx_points: int,
        x_range: List[List[float]],
        num_bins: int = 100,
        device: Union[torch.device, str] = "cpu",
    ):
        assert self.data_batch_size == self.dataset_info["batch_size"]
        # assert n_total_points == self.dataset_info["n_total_points"]
        assert n_ctx_points <= self.dataset_info["max_ctx_points"]
        # assert x_range == self.dataset_info["x_range"]
        # assert num_bins == self.dataset_info["num_bins"]

        # make sure the dataset match
        if batch_size == self.train_batch_size:
            try:
                data = self._get_data(self.train_dataiter)
                return data
            except:
                self.train_dataiter = iter(self.train_dataloader)
                data = self._get_data(self.train_dataiter)
                return data
        elif batch_size == self.eval_batch_size:
            try:
                data = self._get_data(self.eval_dataiter)
                return data
            except:
                self.eval_dataiter = iter(self.eval_dataloader)
                data = self._get_data(self.eval_dataiter)
                return data
        else:
            raise "batch size doesn't match!"

    def load_yaml(self, path):
        with open(f"{path}/dataset.yaml", "r") as file:
            loaded_data = yaml.safe_load(file)
        return loaded_data


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/cfgs", config_name="train")
def generate_offline_data(cfg):
    num_latent = len(cfg.dataset.x_range[0]) + 1
    data_generator_params = {
        "batch_size": cfg.dataset.problem.data_batch_size,
        "n_total_points": cfg.dataset.n_total_points - num_latent,
        "num_latent": cfg.dataset.num_latent,
        "num_bins": cfg.dataset.num_bins,
        "x_range": cfg.dataset.x_range,
        "max_ctx_points": cfg.dataset.max_ctx_points,
        "device": cfg.dataset.device,
    }
    data_generator = OfflineOptDataGenerator(
        cfg.dataset.num_data,
        num_vcpu=cfg.dataset.num_vcpu,
        num_processes=cfg.dataset.num_processes,
        data_dir=cfg.dataset.data_dir,
        **data_generator_params,
    )
    data_generator.run()


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/cfgs", config_name="train")
def test_offline_gen(cfg):
    generate_offline_data(
        cfg.dataset.data_dir,
        cfg.dataset.num_data,
        cfg.dataset.num_vcpu,
        cfg.dataset.num_processes,
        cfg.dataset.x_range,
        cfg.dataset.num_latent,
        cfg.dataset.problem.data_batch_size,
        cfg.dataset.n_total_points,
        cfg.dataset.num_bins,
        cfg.dataset.max_ctx_points,
        cfg.dataset.device,
    )


if __name__ == "__main__":
    if 1:
        with torch.no_grad():
            generate_offline_data()
    if 0:
        # Example usage

        x_range = torch.Tensor([[-1.0], [1.0]])
        data_dir = "./data_bo"
        problem = ProblemWrapper(
            data_dir=data_dir,
            data_batch_size=1024,
            train_batch_size=64,
            eval_batch_size=32,
            eval_split=0.99,
            num_workers=5,
            prior=False,
        )

        data = problem.get_data(
            batch_size=64,
            n_total_points=100,
            n_ctx_points=40,
            x_range=x_range,
            num_bins=100,
            device="cpu",
        )
        breakpoint()
