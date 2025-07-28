import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#
#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         x, y = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.float32))
#         sample = {'image': image, 'label': label.long()}
#         return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Handle different input shapes
        if len(image.shape) == 2:  # Grayscale (H,W)
            image = np.expand_dims(image, -1)  # Convert to (H,W,1)
        elif len(image.shape) == 3 and image.shape[0] <= 3:  # Channel-first (C,H,W)
            image = image.transpose(1, 2, 0)  # Convert to (H,W,C)
        
        # Ensure label is 2D
        if len(label.shape) > 2:
            label = label.squeeze()
        
        # Apply augmentations to all channels
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
            
        # Resize
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            # For color images, zoom each channel separately
            if image.shape[-1] == 3:  # RGB
                zoomed = np.zeros((self.output_size[0], self.output_size[1], 3))
                for c in range(3):
                    zoomed[..., c] = zoom(image[..., c], 
                                        (self.output_size[0]/h, self.output_size[1]/w), 
                                        order=3)
                image = zoomed
            else:  # Grayscale
                image = zoom(image, (self.output_size[0]/h, self.output_size[1]/w), order=3)
            label = zoom(label, (self.output_size[0]/h, self.output_size[1]/w), order=0)
        
        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32))
        if len(image.shape) == 2:  # If grayscale got squeezed
            image = image.unsqueeze(0)  # (1,H,W)
        else:
            image = image.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        label = torch.from_numpy(label.astype(np.float32)).long()
        
        return {'image': image, 'label': label}


# class Watch_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None):
#         self.transform = transform  # using transform in torch!
#         self.split = split
#         self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
#         self.data_dir = base_dir
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def __getitem__(self, idx):
#         if self.split == "train":
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name+'.npz')
#             data = np.load(data_path)
#             image, label = data['image'], data['label']
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#             data = h5py.File(filepath)
#             image, label = data['image'][:], data['label'][:]
#
#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         return sample


class Watch_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        # Read the list file and strip newlines
        list_path = os.path.join(list_dir, f"{split}.txt")
        self.sample_list = [line.strip() for line in open(list_path).readlines()]
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Get filename without extension if present
        case_name = self.sample_list[idx]
        if case_name.endswith('.npz'):
            case_name = case_name[:-4]  # Remove .npz if present in list
            
        # Construct file path
        filepath = os.path.join(self.data_dir, f"{case_name}.npz")
        
        try:
            # Load NPZ file
            with np.load(filepath) as data:
                image = data['image']
                label = data['label']
                
                # Ensure 3D shape for 2D images (C,H,W)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
                
                sample = {'image': image, 'label': label}
                
                if self.transform:
                    sample = self.transform(sample)
                    
                sample['case_name'] = case_name
                return sample
                
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            raise
