import os
import torch
import torch.nn as nn
from dataset import AdvDataset
from torch.utils.data import DataLoader
from advAttack import pgd, pgd_with_restarts
from attack import gen_adv_examples, create_dir
from efficientnet import efficientnet
from defaultSetup import device, root, transform, batch_size
from main import load_model

def main(model_name="efficientnet_b0", num_classes=10):
    """Run only PGD attack and save results."""
    print(f"Using {model_name} for PGD attack.")

    # Load benign dataset for evaluation
    benign_set = AdvDataset(root, transform=transform)
    benign_loader = DataLoader(benign_set, batch_size=batch_size, shuffle=False)

    # Load the trained model
    checkpoint_path = f"./checkpoints/{model_name}_checkpoint.pth"
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Generate and evaluate PGD examples
    print("Running PGD attack...")
    adv_examples, adv_acc, adv_loss, adv_diff = gen_adv_examples(model, benign_loader, pgd_with_restarts, loss_fn)
    print(f'PGD Accuracy: {adv_acc:.5f}, Loss: {adv_loss:.5f}, Perturbation Difference: {adv_diff:.5f}')
    create_dir(root, "pgd", adv_examples, benign_set.__getname__())
    print("PGD attack completed and examples saved to ./pgd/ directory")

if __name__ == "__main__":
    main(model_name="efficientnet_b0", num_classes=10)