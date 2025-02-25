import torch


class RandomAcqRule:
    def __init__(self, x_ranges) -> None:
        self.x_ranges = x_ranges

    def sample(self, **kwargs):
        ndim = self.x_ranges[0].shape[0]
        random_tensor = self.x_ranges[0] + (
            self.x_ranges[1] - self.x_ranges[0]
        ) * torch.rand(1, 1, ndim)
        return random_tensor, None
