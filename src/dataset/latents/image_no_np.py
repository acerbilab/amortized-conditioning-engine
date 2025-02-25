import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class Image(object):
    def __init__(self, rootdir="data", subset="train", load_data=True):
        transforms_list = [transforms.Resize(16), transforms.ToTensor()]
        data = datasets.MNIST(
            root=rootdir,
            train=not (subset == "test"),
            download=load_data,
            transform=transforms.Compose(transforms_list),
        )

        self.data = data
        self.dim_y, _, self.image_size = data[0][0].shape
        self.n_tot = self.image_size**2
        axis = torch.arange(self.image_size, dtype=torch.float32) / (
            self.image_size - 1
        )
        self.grid = torch.stack(
            (axis.repeat_interleave(self.image_size), axis.repeat(self.image_size))
        )

    def get_data(
        self,
        batch_size=16,
        n_total_points=256,
        n_ctx_points=None,
        x_range=None,
        device="cpu",
    ):
        inds = torch.randint(0, len(self.data), (batch_size,))
        features = torch.cat([self.data[ind][0] for ind in inds])
        labels = torch.tensor([self.data[ind][1] for ind in inds])

        # target_features
        target_x = self.grid.repeat([batch_size, 1, 1])
        target_y = features.reshape(batch_size, self.dim_y, -1)

        # this is needed so that data values do not match a discrete label value?
        target_y = torch.maximum(target_y, torch.tensor(0.0000001))
        target_y = torch.minimum(target_y, torch.tensor(0.9999999))

        # random subset
        xd = target_x.permute(0, 2, 1)
        yd = target_y.permute(0, 2, 1)
        d_index = torch.argsort(torch.rand((batch_size, self.n_tot)), dim=-1)[
            :, :n_total_points
        ]
        xd_rand = torch.gather(
            xd, dim=1, index=d_index.to(torch.long).unsqueeze(-1).tile(2)
        )
        yd_rand = torch.gather(yd, dim=1, index=d_index.to(torch.long).unsqueeze(-1))
        data_marker = torch.full_like(yd_rand, 1)
        batch_xyd = torch.cat(
            (data_marker, xd_rand, yd_rand), dim=-1
        )  # batch size x num points x 4

        # latent
        batch_xyl = torch.zeros((batch_size, 1, 4))  # batch size x num latent x 4
        batch_xyl[:, 0, 0] = 2
        batch_xyl[:, 0, 3] = labels

        return batch_xyd, batch_xyl


if __name__ == "__main__":
    image = Image()
    batch_xyd, batch_xyl = image.get_data()
    ind = 0
    xc = (15 * batch_xyd[ind, :, 1:3]).to(torch.long)
    yc = batch_xyd[ind, :, -1]
    image_size = 16
    im_context = np.zeros((image_size, image_size, 3))
    im_context[:, :, 2] = 1
    im_context[xc[:, 0], xc[:, 1]] = yc.repeat(3, 1).transpose(0, 1)
    plt.imshow(im_context)
    plt.show()
