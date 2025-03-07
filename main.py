import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import AdvDataset
from torch.utils.data import DataLoader
from advAttack import fgsm, pgd, ifgsm
from attack import gen_adv_examples, create_dir
from resnet import resnet50
from efficientnet import efficientnet
from defaultSetup import device, root, transform, batch_size

# Model selection dictionary
MODEL_DICT = {
    "resnet50": lambda num_classes: resnet50(pretrained=True, num_classes=num_classes).to(device),
    "efficientnet_b0": lambda num_classes: efficientnet("efficientnet_b0", pretrained=True, num_classes=num_classes).to(device),
    "efficientnet_b3": lambda num_classes: efficientnet("efficientnet_b3", pretrained=True, num_classes=num_classes).to(device)
}

def train_model(model, train_loader, loss_fn, optimizer, epochs=10, save_path="model_checkpoint.pth"):
    """Train the model for a given number of epochs and save the checkpoint."""
    model.train()
    
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved at {save_path}")

    return model  # Return trained model to avoid reloading

def load_model(model_name, checkpoint_path, num_classes):
    """Load a trained model for evaluation."""
    if model_name not in MODEL_DICT:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_DICT.keys())}")

    model = MODEL_DICT[model_name](num_classes)
    
    # Load the trained weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Using untrained model.")

    model.eval()
    return model

def evaluate_model(model, loader, loss_fn, label="Benign"):
    """Evaluate the model on benign or adversarial data."""
    model.eval()
    correct, total_loss, total = 0, 0.0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)

            total_loss += loss.item() * x.shape[0]
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total
    print(f'{label} Accuracy: {accuracy:.5f}, Loss: {avg_loss:.5f}')
    return accuracy, avg_loss

def main(model_name="resnet50", num_classes=10, epochs=20, lr=0.001):
    """Main function to train and evaluate a model."""
    print(f"Using {model_name} for training and evaluation.")

    # Load dataset
    train_set = AdvDataset(root, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Separate benign dataset for evaluation
    benign_set = AdvDataset(root, transform=transform)  # Load benign dataset separately if needed
    benign_loader = DataLoader(benign_set, batch_size=batch_size, shuffle=False)

    # Initialize model
    if model_name not in MODEL_DICT:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_DICT.keys())}")

    model = MODEL_DICT[model_name](num_classes)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    checkpoint_path = f"./checkpoints/{model_name}_checkpoint.pth"
    print(f"Training {model_name} for {epochs} epochs...")
    model = train_model(model, train_loader, loss_fn, optimizer, epochs=epochs, save_path=checkpoint_path)

    # Load trained model for evaluation
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)

    # Evaluate on benign examples
    benign_acc, benign_loss = evaluate_model(model, benign_loader, loss_fn, label="Benign")

    # Generate and evaluate adversarial examples
    attacks = {"fgsm": fgsm, "ifgsm": ifgsm, "pgd": pgd}
    for attack_name, attack_fn in attacks.items():
        adv_examples, adv_acc, adv_loss, adv_diff = gen_adv_examples(model, benign_loader, attack_fn, loss_fn)
        print(f'{attack_name.upper()} Accuracy: {adv_acc:.5f}, Loss: {adv_loss:.5f}, Perturbation Difference: {adv_diff:.5f}')
        create_dir(root, attack_name, adv_examples, train_set.__getname__())

if __name__ == "__main__":
    # Change model here by setting model_name
    main(model_name="efficientnet_b0", num_classes=10, epochs=20, lr=0.001)
