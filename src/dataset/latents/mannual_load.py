import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
# CelebA https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&export=download&authuser=0


class CelebADataset(Dataset):
    def __init__(self, root_dir="code/data/celeba/img_align_celeba",
                 attr_path="code/data/celeba/list_attr_celeba.txt", transform=None,
                 feature_index = ['Black_Hair', 'Gray_Hair', 'Brown_Hair', 'Blond_Hair', 'Bald'] ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.feature_index = feature_index

        # root_dir = "code/data/celeba/img_align_celeba"
        # attr_path="code/data/celeba/list_attr_celeba.txt"

        image_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        df = pd.read_csv(attr_path, delim_whitespace=True, header=1, index_col=0)
        # df = df[self.feature_index]
        # df[df == -1] = 0
        # s_series = df.sum(axis=1)
        # df2 = df[(s_series > 1)]
        # index_multi = df2.index.to_list()

        # self.image_files = [x for x in image_all if x not in index_multi]
        # df_hair = df.drop(index_multi)

        # # Create a hair color map
        # hair_color_map = {feature: index + 1 for index, feature in enumerate(feature_index)}
        # df_new = pd.DataFrame(index=df.index)
        # conditions = [df[color] == 1 for color in hair_color_map.keys()] #list of series for each feature
        # choices = list(hair_color_map.values())

        # df_new['Hair_Color_Class'] = np.select(conditions, choices, default=0) # returns number if binary=1
        # self.attributes = df_new
        # self.full_attributes = df_hair
        # self.color_map = hair_color_map

        #### ALL FEATURES
        #index_multi = df.index.to_list()
        self.image_files = [x for x in image_all]
        df[df == -1] = 0

        # We will iterate over the columns and apply the transformation
        for i, col in enumerate(df.columns):
            df[col] = df[col].apply(lambda x: 2 * i if x == 0 else 2 * i + 1)


        self.attributes = df


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.image_files[idx])
            image = Image.open(img_name).convert('RGB')

            # Check the image size before applying transformations
            if image.size != (178, 218):
                raise ValueError(f"Image {img_name} has unexpected size {image.size}, skipping.")

            # Apply transformations if the image size is correct
            if self.transform:
                image = self.transform(image)
            
            # Convert attributes to a tensor
            attrs = torch.tensor(self.attributes.loc[self.image_files[idx]].values.astype(int))
            
            return image, attrs

        except Exception as e:
            # Print the specific error along with the image name
            print(f"Error loading image {img_name}: {e}")
            # Skip the problematic image by returning the next image
            return self.__getitem__((idx + 1) % len(self))

    
        # img_name = os.path.join(self.root_dir, self.image_files[idx])
        # image = Image.open(img_name).convert('RGB')  # Convert to RGB
        # print(f"Image shape after transform: {image.shape}")  # Should be [3, 32, 32]

        # attrs = torch.tensor(self.attributes.loc[self.image_files[idx]].values.astype(int))

        # if self.transform:
        #     image = self.transform(image)

        # # image now has shape [3, 32, 32] after transformations
        # return image, attrs



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_images(images, attributes, filenames):
        # Determine the number of images
        num_images = len(images)
        
        # Create a subplot with appropriate size
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        
        # Check if we have more than one image to avoid indexing issues
        if num_images == 1:
            axes = [axes]  # Encapsulate in a list to make iterable

        for ax, img, attr, filename in zip(axes, images, attributes, filenames):
            ax.imshow(img.squeeze(0))  # Convert tensor image to HWC format for plotting

            ax.set_title(filename)
            # Use the space below the image to list attributes
            ax.set_xlabel('\n'.join(str(attr.numpy())), fontsize=9)  # Set attributes as the x-label in smaller font size
            ax.xaxis.set_label_position('bottom')  # Position the label at the bottom
            ax.xaxis.tick_bottom()  # Ensure ticks are at the bottom
            ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks/length
            ax.set_xticks([])  # Clear x-axis ticks (image indices not needed)
            ax.set_yticks([])  # Clear y-axis ticks
            #ax.axis('off')  # Hide the axis border

        plt.tight_layout()  # Adjust layout to not overlap
        plt.show()

    torch.manual_seed(10)


    # Define transformations
    transforms_list = transforms.Compose([
        transforms.Resize(32),  # Resize the image to 64x64
        transforms.CenterCrop(32),  # Crop the images to 64x64
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5066832 , 0.4247095 , 0.38070202], std=[0.30913046, 0.28822428, 0.2866247])  # Normalize images
    ])
    hair_index = ["Black_Hair", 'Gray_Hair', 'Brown_Hair', 'Blond_Hair', 'Bald']

    celeba_dataset = CelebADataset(root_dir="code/data/celeba/img_align_celeba",
                                attr_path="code/data/celeba/list_attr_celeba.txt",
                                transform=transforms_list, feature_index=hair_index)

    from torch.utils.data import DataLoader

    # Assuming `celeba_dataset` is already created as shown above
    batch_size = 8  # You can adjust this number based on your requirement

    # Create the DataLoader for batching images
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)

    # Get one batch of data
    images, attrs = next(iter(data_loader))

    # Get filenames for this batch (assuming each batch aligns with filenames)
    filenames = [celeba_dataset.image_files[i] for i in range(len(images))]

    # Now plot the images with attributes

    plot_images(images, attrs, filenames) # This needs fixing


