from torch.utils.data import Dataset, DataLoader
import torch


class SBIDataset(Dataset):
    def __init__(self, x_file, theta_file, order="random"):
        self.X = torch.load(x_file)  # [num_samples, num_points]
        self.theta = torch.load(theta_file)  # [num_samples, 2]
        self.order = order
        self.num_samples = self.X.size(0)
        self.num_points = self.X.size(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X_sample = self.X[idx]  # [num_points]
        theta_sample = self.theta[idx]

        theta_dim = theta_sample.size(0)

        if self.order == "random":
            d_index = torch.randperm(self.num_points)
            xd = (
                torch.arange(self.num_points).unsqueeze(-1).float()[d_index]
            )  # [num_points, 1]
            yd = X_sample.unsqueeze(-1)[d_index]  # [num_points, 1]
        else:
            xd = torch.arange(self.num_points).unsqueeze(-1).float()
            yd = X_sample.unsqueeze(-1)

        # xyd: [num_points, 3]，[1, x, y]
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        # xyl: [theta_dim, 3]，[latent_marker, xl, yl]
        xl = torch.zeros(theta_dim, 1).float()
        yl = theta_sample.unsqueeze(-1).float()  # [theta_dim, 1]

        latent_marker = torch.arange(2, theta_dim + 2).unsqueeze(-1).float()
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xyd, xyl


class SBIDatasetPI(Dataset):
    def __init__(self, x_file, theta_file, weights_file, order="random"):
        self.X = torch.load(x_file)  # [num_samples, num_points]
        self.theta = torch.load(theta_file)  # [num_samples, theta_dim]
        self.weights = torch.load(weights_file)  # [num_samples, theta_dim, 100]
        self.order = order
        self.num_samples = self.X.size(0)
        self.num_points = self.X.size(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X_sample = self.X[idx]  # [num_points]
        theta_sample = self.theta[idx]  # [theta_dim]
        weights_sample = self.weights[idx]  # [theta_dim, 100]

        theta_dim = theta_sample.size(0)

        if self.order == "random":
            d_index = torch.randperm(self.num_points)
            xd = (
                torch.arange(self.num_points).unsqueeze(-1).float()[d_index]
            )  # [num_points, 1]
            yd = X_sample.unsqueeze(-1)[d_index]  # [num_points, 1]
        else:
            xd = torch.arange(self.num_points).unsqueeze(-1).float()
            yd = X_sample.unsqueeze(-1)

        # xyd: [num_points, 3]，[1, x, y]
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        # xyl: [theta_dim, 3]，[latent_marker, xl, yl]
        xl = torch.zeros(theta_dim, 1).float()
        yl = theta_sample.unsqueeze(-1).float()  # [theta_dim, 1]
        yl_weights = weights_sample
        latent_marker = torch.arange(2, theta_dim + 2).unsqueeze(-1).float()
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)

        return xyd, xyl


class SBIDatasetPINew(Dataset):
    """
    New dataset works with the new prior sampler.
    """

    def __init__(self, x_file, theta_file, weights_file, order="random"):
        self.X = torch.load(x_file)  # [num_samples, num_points]
        self.theta = torch.load(theta_file)  # [num_samples, theta_dim]
        self.weights = torch.load(weights_file)  # [num_samples, theta_dim, 100]
        self.order = order
        self.num_samples = self.X.size(0)
        self.num_points = self.X.size(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X_sample = self.X[idx]  # [num_points]
        theta_sample = self.theta[idx]  # [theta_dim]
        weights_sample = self.weights[idx]  # [theta_dim, 100]

        theta_dim = theta_sample.size(0)

        if self.order == "random":
            d_index = torch.randperm(self.num_points)
            xd = (
                torch.arange(self.num_points).unsqueeze(-1).float()[d_index]
            )  # [num_points, 1]
            yd = X_sample.unsqueeze(-1)[d_index]  # [num_points, 1]
        else:
            xd = torch.arange(self.num_points).unsqueeze(-1).float()
            yd = X_sample.unsqueeze(-1)

        # xyd: [num_points, 3]，[1, x, y]
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        # xyl: [theta_dim, 3]，[latent_marker, xl, yl]
        xl = torch.zeros(theta_dim, 1).float()
        yl = theta_sample.unsqueeze(-1).float()  # [theta_dim, 1]
        yl_weights = weights_sample
        latent_marker = torch.arange(2, theta_dim + 2).unsqueeze(-1).float()
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xyd, xyl, yl_weights


class SBILoader(object):
    def __init__(
        self, x_file, theta_file, batch_size=100, shuffle=True, order="random"
    ):
        self.x_file = x_file
        self.theta_file = theta_file
        self.dataset = SBIDataset(self.x_file, self.theta_file, order=order)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        self.dataiter = iter(self.dataloader)

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=None,
        x_range=None,
        device="cpu",
    ):
        try:
            xyd, xyl = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            xyd, xyl = next(self.dataiter)
        return xyd, xyl


class SBILoaderPI(object):
    def __init__(
        self,
        x_file,
        theta_file,
        weights_file,
        batch_size=100,
        shuffle=True,
        order="random",
    ):
        self.x_file = x_file
        self.theta_file = theta_file
        self.weights_file = weights_file
        self.dataset = SBIDatasetPI(
            self.x_file, self.theta_file, self.weights_file, order=order
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        self.dataiter = iter(self.dataloader)

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=None,
        x_range=None,
        num_bins=None,
        device="cpu",
    ):
        try:
            xyd, xyl = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            xyd, xyl = next(self.dataiter)
        return xyd, xyl


class SBILoaderPINew(object):
    def __init__(
        self,
        x_file,
        theta_file,
        weights_file,
        batch_size=100,
        shuffle=True,
        order="random",
    ):
        self.x_file = x_file
        self.theta_file = theta_file
        self.weights_file = weights_file
        self.dataset = SBIDatasetPINew(
            self.x_file, self.theta_file, self.weights_file, order=order
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        self.dataiter = iter(self.dataloader)

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=None,
        x_range=None,
        num_bins=None,
        device="cpu",
    ):
        try:
            xyd, xyl, weights = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            xyd, xyl, weights = next(self.dataiter)
        return xyd, xyl, weights
