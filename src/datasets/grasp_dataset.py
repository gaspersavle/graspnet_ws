import os
import numpy as np
import torch
from torch.utils.data import Dataset

class GraspDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        data = np.load(path)

        image = data["image"].astype(np.float32) / 255.0  # Normalize to [0,1]
        label = data["label"].astype(np.int64)
        heatmap = data["heatmaps"].astype(np.float32).max(axis=-1, keepdims=True)  # [H, W, 1]
        orientation = data["orientation"].astype(np.float32)  # [2, H, W]

        # Convert to CHW
        image = torch.from_numpy(image.transpose(2, 0, 1))       # [3, H, W]
        label = torch.from_numpy(label)                          # [H, W]
        heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))   # [1, H, W]
        orientation = torch.from_numpy(orientation)              # [2, H, W]

        return {
            "image": image,
            "label": label,
            "grasp_heatmap": heatmap,
            "orientation": orientation
        }

