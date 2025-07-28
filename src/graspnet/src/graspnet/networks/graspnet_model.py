# graspnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspNet(nn.Module):
    def __init__(self, input_channels=29):  # 3 + 26 classes
        super(GraspNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.grasp_heatmap_head = nn.Conv2d(64, 1, 1)           # Graspability map
        self.orientation_head = nn.Conv2d(64, 2, 1)             # Cos(\theta), Sin(\theta)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        heatmap = torch.sigmoid(self.grasp_heatmap_head(features))
        orientation = F.normalize(self.orientation_head(features), dim=1)
        confidence = self.confidence_head(features)
        return heatmap, orientation, confidence

