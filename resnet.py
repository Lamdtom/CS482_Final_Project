import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet50, self).__init__()
        
        # Load ResNet-50 with optional pretrained weights
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50()

        # Replace fully connected (fc) layer to match num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to initialize the model
def resnet50(pretrained=False, num_classes=10):
    return ResNet50(num_classes=num_classes, pretrained=pretrained)

# Test the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True).to(device)  # Load with pretrained weights
    print(model)  # Print model architecture
import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet50, self).__init__()
        
        # Load ResNet-50 with optional pretrained weights
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50()

        # Replace fully connected (fc) layer to match num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to initialize the model
def resnet50(pretrained=False, num_classes=10):
    return ResNet50(num_classes=num_classes, pretrained=pretrained)

# Test the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True).to(device)  # Load with pretrained weights
    print(model)  # Print model architecture
