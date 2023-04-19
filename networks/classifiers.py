import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

class LinearClassifier(nn.Module):
    def __init__(self, num_classes=None):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(256, num_classes)
    def forward(self, features):
        return self.fc(features)


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=None, feature_dim=256, type="linear"):
        super(LinearClassifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(feature_dim, num_classes)
        else:
            self.fc = weightNorm(nn.Linear(feature_dim, num_classes), name="weight")

    def forward(self, features):
        return self.fc(features)
