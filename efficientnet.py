import torch
import torch.nn as nn
from torchvision import models

class EfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", num_classes=10, pretrained=True):
        super(EfficientNet, self).__init__()
        # Load pre-trained EfficientNet model
        self.model = getattr(models, model_name)(pretrained=pretrained)
        
        # Modify the final classifier layer to match num_classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to instantiate the model
def efficientnet(model_name="efficientnet_b0", num_classes=10, pretrained=True):
    return EfficientNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet("efficientnet_b0", num_classes=10, pretrained=True).to(device)
    print(model)  # Display model architecture
