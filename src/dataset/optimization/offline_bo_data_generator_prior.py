from typing import List, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from src.dataset.optimization.bo_data_generator_prior import (
    BayesianOptimizationDataGeneratorPrior,
)
from torch.utils.data import DataLoader, random_split

import hydra
import time
import math
import yaml
import os
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
    ):
        self.num_data = num_data
        self.num_processes = num_processes
        self.mp_method = mp_method or "spawn"
        self.verbose = verbose
        self.num_threads = max(1, math.floor(num_vcpu / num_processes))
        self.data_dir = data_dir

        # Initialize data generation parameters
        self.batch_size = data_generator_params.get("gen_batch_size")
        self.n_total_points = data_generator_params.get("n_total_points")
        self.num_latent = data_generator_params.get("num_latent")
        self.max_ctx_points = data_generator_params.get("max_ctx_points")
        self.num_bins = data_generator_params.get("num_bins")
        self.x_range = data_generator_params.get("x_range", [])
        self.device = data_generator_params.get("device")
        self.prior_type = data_generator_params.get("prior_type")
        self.data_yaml = os.path.join(data_dir, "dataset.yaml")

        # Convert x_range to list and save parameters to yaml
        data_generator_params["x_range"] = torch.tensor(self.x_range).tolist()
        os.makedirs(data_dir, exist_ok=True)
        self._save_params_to_yaml(data_generator_params)

    def _save_params_to_yaml(self, params):
        with open(self.data_yaml, "w") as file:
            yaml.dump(params, file)

    def run(self):
        for i_rank in range(0, self.num_data, self.num_processes):
            self.parallel_run(i_rank)

        if self._has_sufficient_files():
            self.gather_files(self.data_dir, self.num_data)
        else:
            raise ValueError(f"Insufficient files in {self.data_dir}")

    def _has_sufficient_files(self):
        file_count = len(
            [
                f
                for f in os.listdir(self.data_dir)
                if os.path.isfile(os.path.join(self.data_dir, f))
            ]
        )
        return file_count >= self.num_data + 1

    def parallel_run(self, i_rank):
        start_time = time.time() if self.verbose else None
        processes = []

        for rank in range(i_rank, i_rank + self.num_processes):
            seed = torch.initial_seed() + rank
            process = mp.Process(target=self.worker_process, args=(rank, seed))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        if self.verbose:
            elapsed_time = time.time() - start_time
            print(
                f"All processes completed and data saved in {elapsed_time:.2f} seconds"
            )

    def worker_process(self, rank, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        torch.set_num_threads(self.num_threads)

        if self.verbose:
            print(f"Process {rank} started")

        data = self.get_data()

        if self.verbose:
            print(f"Process {rank} finished")

        file_name = os.path.join(
            self.data_dir, f"optim{self.num_latent - 1}D_{rank + 1}_data.pt"
        )
        torch.save(data, file_name)
        return data

    def _get_data(self):
        """Generate data using BayesianOptimizationDataGeneratorPrior."""

        dataset = BayesianOptimizationDataGeneratorPrior()

        batch_xyd, batch_xyl, latent_bin_weight = dataset.get_data(
            batch_size=self.batch_size,
            n_total_points=self.n_total_points - self.num_latent,
            n_ctx_points=self.max_ctx_points,
            num_bins=self.num_bins,
            x_range=self.x_range,
            device=self.device,
            prior_type=self.prior_type,
        )
        return batch_xyd, batch_xyl, latent_bin_weight

    def get_data(self):
        attempts = 0
        while attempts < MAX_SAMPLE_RETRIES:
            attempts += 1
            try:
                batch_xyd, batch_xyl, latent_bin_weight = self._get_data()
                self.check_tensors(batch_xyd, batch_xyl, latent_bin_weight)
                return batch_xyd, batch_xyl, latent_bin_weight
            except RuntimeError as e:
                print(f"Attempt {attempts} failed with error: {e}")
                if attempts >= MAX_SAMPLE_RETRIES:
                    print("Max retries reached. Aborting.")
                    raise e

    def check_tensors(self, *tensors):
        for tensor in tensors:
            if torch.isnan(tensor).sum() > 0:
                raise ValueError(f"Tensor contains NaN values: {tensor}")
            if torch.isinf(tensor).sum() > 0:
                raise ValueError(f"Tensor contains infinite values: {tensor}")

    def gather_files(self, path, n_files):
        files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])[:n_files]
        concatenated_tensors = [[], [], []]

        for file in files:
            file_path = os.path.join(path, file)
            tensor_tuple = torch.load(file_path)
            assert (
                len(tensor_tuple) == 3
            ), f"File {file} does not contain a tuple of size 3"

            for i in range(3):
                concatenated_tensors[i].append(tensor_tuple[i])

        concatenated_tensors = [
            torch.cat(tensors, dim=0) for tensors in concatenated_tensors
        ]
        output_file = os.path.join(path, "tensors.pt")
        torch.save(tuple(concatenated_tensors), output_file)

        self._cleanup_files(path, keep_files={"tensors.pt", "dataset.yaml"})

    def _cleanup_files(self, path, keep_files):
        # reread for parallel behaviour hack
        files = os.listdir(path)

        for file in files:
            if file not in keep_files:
                os.remove(os.path.join(path, file))


class PriorDataset(Dataset):
    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, file_name)
        self.tensor = torch.load(self.file_path)

    def __len__(self):
        return len(self.tensor[1])

    def __getitem__(self, idx):
        xyd, xyl, bin_weight = self.tensor
        return xyd[idx], xyl[idx], bin_weight[idx]


class NoPriorDataset(PriorDataset):
    def __init__(self, data_dir, file_name):
        super().__init__(data_dir, file_name)

    def __getitem__(self, idx):
        xyd, xyl, _ = self.tensor
        return xyd[idx], xyl[idx]


class ProblemWrapper:
    """
    A wrapper class for handling dataset loading, splitting, and batching for training and evaluation.

    Attributes:
        dataset_info (dict): Information about the dataset loaded from a YAML file.
        data_dir (str): Directory where the dataset is stored.
        train_batch_size (int): Batch size for training data.
        eval_batch_size (int): Batch size for evaluation data.
        train_dataloader (DataLoader): DataLoader for training data.
        eval_dataloader (DataLoader): DataLoader for evaluation data.
        train_dataiter (iterator): Iterator for training DataLoader.
        eval_dataiter (iterator): Iterator for evaluation DataLoader.

    Methods:
        _get_data(data_iter):
            Retrieves the next batch of data from the given iterator.

        get_data(batch_size, n_total_points, n_ctx_points, x_range, num_bins=100, device="cpu"):
            Retrieves a batch of data based on the specified parameters.

        load_yaml(path):
            Loads dataset information from a YAML file at the given path.
    """

    def __init__(
        self,
        data_dir,
        train_batch_size,
        eval_batch_size,
        eval_split,
        train_test_split=0.9,
        num_workers=4,
        prior=True,
        file_name="tensors.pt",
    ) -> None:
        self.dataset_info = self.load_yaml(data_dir)
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size / eval_split

        if prior:
            dataset = PriorDataset(self.data_dir, file_name)
        else:
            dataset = NoPriorDataset(self.data_dir, file_name)

            # Define split ratios
        train_size = int(train_test_split * len(dataset))  # 80% training
        eval_size = len(dataset) - train_size  # 20% test
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=True,
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
        assert n_ctx_points <= self.dataset_info["max_ctx_points"]

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
        "gen_batch_size": cfg.dataset.gen_batch_size,
        "batch_size": cfg.batch_size,
        "n_total_points": cfg.dataset.n_total_points - num_latent,
        "num_latent": cfg.dataset.num_latent,
        "num_bins": cfg.dataset.num_bins,
        "x_range": cfg.dataset.x_range,
        "max_ctx_points": cfg.dataset.max_ctx_points,
        "device": cfg.dataset.device,
        "prior_type": cfg.dataset.prior_type,
    }
    data_generator = OfflineOptDataGenerator(
        cfg.dataset.num_data,
        num_vcpu=cfg.dataset.num_vcpu,
        num_processes=cfg.dataset.num_processes,
        data_dir=cfg.dataset.data_dir,
        **data_generator_params,
    )
    data_generator.run()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    if 1:
        # test generate
        with torch.no_grad():
            generate_offline_data()
    if 0:
        # test sampler

        # build sampler
        sampler = ProblemWrapper(
            data_dir="./offline_data/test/",
            train_batch_size=64,
            eval_batch_size=20,
            eval_split=1,
            train_test_split=0.9,
        )

        # getting train sample batch_size must match
        train_sample = sampler.get_data(
            batch_size=64,
            n_total_points=100,
            n_ctx_points=20,
            x_range=[[-1], [1]],
        )

        # getting eval sample batch_size must match
        eval_sample = sampler.get_data(
            batch_size=20,
            n_total_points=100,
            n_ctx_points=20,
            x_range=[[-1], [1]],
        )
        if 0:
            fig, axs = plt.subplots(
                4, 4, figsize=(12, 12)
            )  # Create a 4x4 grid of subplots
            axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array

            for i in range(16):
                bins = eval_sample[-1][i, 0, :].detach()
                xopt = eval_sample[1][i, 0, -1]
                ax = axs[i]  # Get the correct subplot
                ax.plot(np.linspace(-1, 1, len(bins)), bins)
                ax.vlines(xopt, ymin=bins.min(), ymax=bins.max(), color="r")
                ax.set_title(
                    f"xopt: {xopt.item():.2f}"
                )  # Update the title with xopt value

            # Hide unused subplots if necessary
            for j in range(i + 1, len(axs)):
                axs[j].axis("off")

            plt.tight_layout()  # Adjust the layout to fit subplots
            plt.show()

    """
    #offline_gen_test.yaml
    num_data : 1000 # n functions sampled is: this times gen_batch_size
    gen_batch_size: 1024 #batch size during paralel generation
    num_vcpu : 3
    num_processes : 3
    data_dir : "./offline_data/test/"
    mp_method: "spawn"
    _target_: src.dataset.ctx_tar_sampler_prior.ContextTargetSamplerWithPrior
    name : "bo_1d_prior"
    num_ctx: 'random'
    dim_input: 2 # 1 + actual input dim
    num_latent: 2 # x1_star, y_star 
    max_ctx_points: 50
    min_ctx_points: 3
    n_total_points: 100 # at least num_latent + max_ctx_points
    x_range: [[-1],[1]]
    loss_latent_weight: 1
    loss_data_weight: 1
    device: 'cpu'
    ctx_tar_sampler: 'bernuniformsampler'
    num_bins: 100
    problem: 
    _target_: src.dataset.optimization.offline_bo_data_generator_prior.ProblemWrapper
    data_dir: ${dataset.data_dir}
    train_batch_size: ${batch_size}
    eval_batch_size: ${eval_batch_size}
    eval_split: ${eval_split}
    train_test_split: 0.9
    num_workers: 9
    prior: True
    """
