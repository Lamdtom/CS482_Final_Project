import torch
import torchvision
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations (normalize with CIFAR-10 statistics)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201))
])

# Download CIFAR-10 dataset
root = './data'  # Directory for storing dataset
trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

# Check dataset size
print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")

# Get a batch of images
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"Batch shape: {images.shape}, Labels: {labels}")
