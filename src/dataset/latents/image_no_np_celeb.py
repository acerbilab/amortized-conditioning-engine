import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from src.dataset.latents.mannual_load import CelebADataset
from torch.utils.data import DataLoader
from time import perf_counter as time
from pathlib import Path


class Image(object):
    
    def __init__(self, rootdir="data", subset="train", load_data=True, celeba=True, batch_size:int=6,num_workers:int=32):

        if celeba:
            transforms_list = transforms.Compose([
                transforms.Resize(64),  # Resize the image to 64x64
                transforms.CenterCrop(64),  # Crop the images to 64x64
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5066832 , 0.4247095 , 0.38070202],
                                        std=[0.30913046, 0.28822428, 0.2866247]),  # Add a comma after the last element
            ])

            hair_index = ["Black_Hair", 'Gray_Hair', 'Brown_Hair', 'Blond_Hair', 'Bald']

            data =  CelebADataset(root_dir="data/celeba/img_align_celeba",
                                attr_path="data/celeba/list_attr_celeba.txt",
                                transform=transforms_list, feature_index=hair_index)
        else:
            transforms_list = [transforms.Resize(16), transforms.ToTensor()]
            data = datasets.MNIST(
                root=rootdir,
                train=not(subset == "test"),
                download=load_data,
                transform=transforms.Compose(transforms_list)
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = data
        self.dataloader_iter = iter(DataLoader(data,num_workers=self.num_workers,
                                                persistent_workers=True,
                                                batch_size=self.batch_size,
                                                shuffle=True, 
                                                drop_last=True))

        if celeba:
                self.image_size = 64 
                self.dim_y = 3
        else:
            self.dim_y, _, self.image_size = data[0][0].shape

        self.n_tot = self.image_size**2
        axis = torch.arange(self.image_size, dtype=torch.float32)/(self.image_size-1)
        # grid of image coordinates
        self.grid = torch.stack((axis.repeat_interleave(self.image_size), axis.repeat(self.image_size)))


    def stream_dataloader(self):
        while True:
            try:
                # Fetch the next batch of data
                features, labels = next(self.dataloader_iter)
                return features, labels  # Return the batch when available
            
            except StopIteration:
                pass

            # If the DataLoader is exhausted, reset the iterator
            self.dataloader_iter = iter(DataLoader(
                self.data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=True,
                drop_last=True,  # Continue streaming but drop incomplete batches
            ))

    def get_data(self,
                n_total_points=4096,
                n_ctx_points=None,
                x_range=None,
                device="cpu"
    ):


        features, labels = self.stream_dataloader()
        #assert features.shape[1] == 3 and features.shape[2] == 32 and features.shape[3] == 32, "Images are not 32x32"

        # features now have shape [batch_size, 3, 32, 32]
        # No need to squeeze
        labels = labels.squeeze(1)  # If labels have an extra dimension

        actual_batch_size = features.size(0)  # Get the actual batch size
        
        target_x = self.grid.repeat([actual_batch_size, 1, 1])  # Shape: [batch_size, 2, 1024]
        target_y = features.reshape(actual_batch_size, self.dim_y, self.n_tot)  # Shape: [batch_size, 3, 1024]

        #assert target_y.shape[2] == 1024, f"Expected 1024 pixels, got {target_y.shape[2]}"
        
        # Permute to get yd of shape [batch_size, 1024, 3]
        yd = target_y.permute(0, 2, 1)  # batch_size x num_points x 3
            
        xd = target_x.permute(0, 2, 1)  # batch_size x num_points x 2

        # load_data_time = time() - start
        # print(f"Loading images took: {load_data_time:.4f} seconds")
        

        d_index = torch.argsort(torch.rand((self.batch_size, self.n_tot)), dim=-1) #[:, :self.n_totn_total_points] index for random subset
        xd_rand = torch.gather(xd, dim=1, index=d_index.to(torch.long).unsqueeze(-1).tile(2))
        # yd_rand = torch.gather(yd, dim=1, index=d_index.to(torch.long).unsqueeze(-1))
        d_index_unsqueezed = d_index.unsqueeze(-1) # Expands to [Batch, 1, 1]

        # print(f"yd.shape: {yd.shape}")
        # print(f"d_index.shape: {d_index.shape}")
        # print(f"Max index in d_index: {torch.max(d_index)}")

        yd_rand = torch.gather(yd, dim=1, index=d_index_unsqueezed.expand(-1, -1, 3)) # Expands index to [Batch, 1, 3] THIS makes the y random also

        data_marker = torch.full_like(xd[:,:,:1], 1)
        #batch_xyd = torch.cat((data_marker, xd, yd), dim=-1)  # batch size x num points x 4
        batch_xyd = torch.cat((data_marker, xd_rand, yd_rand), dim=-1)  # batch size x num points x 4
        # batch_xyd is [data_marker, x_d=2, y_d=3]
        # latent
        #batch_xyl = torch.zeros((self.batch_size, 1, 6))  # batch size x num latent x 4
        # Create a tensor of shape [128, 40, 6] initialized to zeros
        batch_xyl = torch.zeros((self.batch_size, 40, 6))

        # Set the first column of the third dimension (i.e., batch_xyl[:, :, 0]) to values from 2 to 41
        batch_xyl[:, :, 0] = torch.arange(2, 42).unsqueeze(0)  # Values 2 to 41
        #batch_xyl[:, 0, 0] = 2
        batch_xyl[:, :, 5] = labels
        
        return batch_xyd, batch_xyl
    
if __name__ == "__main__":
    image = Image()
    batch_xyd, batch_xyl = image.get_data()
    ind = 0
    xc = (31 * batch_xyd[ind, :, 1:3]).to(torch.long)  # Scale to 32x32
    yc = batch_xyd[ind, :, -3:]  # Get all 3 channels
    image_size = 32
    
    # Create the original image
    im_original = torch.zeros((image_size, image_size, 3))
    im_original[xc[:, 0], xc[:, 1]] = yc
    
    # Create the masked image
    im_masked = im_original.clone()
    im_masked[:image_size//2, :, :] = 0.5  # Set top half to gray (0.5)
    
    # Plot both images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(im_original)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(im_masked)
    ax2.set_title("Masked Image")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
