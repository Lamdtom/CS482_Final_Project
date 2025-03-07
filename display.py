import matplotlib.pyplot as plt
from PIL import Image
from defaultSetup import transform, device
import numpy as np
from resnet import resnet50
from efficientnet import efficientnet
from main import load_model

# Define class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Model selection dictionary

# Function to load selected model
def load_selected_model(model_name='efficientnet_b0', checkpoint_path="model_checkpoint.pth", num_classes=10):
    return load_model(model_name=model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)

# Select the model you want to load
selected_model = 'efficientnet_b0'  # Change to 'resnet50' or 'efficientnet_b3' for other models
checkpoint_path = f"./checkpoints/{selected_model}_checkpoint.pth"
# Load the model dynamically
model = load_selected_model(selected_model, checkpoint_path=checkpoint_path)

# Create a figure with 2 columns (Benign | Adversarial)
fig, axes = plt.subplots(len(classes), 2, figsize=(8, 20))

for i, cls_name in enumerate(classes):
    # Benign Image
    benign_path = f'./data/{cls_name}/{cls_name}1.png'
    benign_im = Image.open(benign_path)
    benign_logit = model(transform(benign_im).unsqueeze(0).to(device))[0]
    benign_pred = benign_logit.argmax(-1).item()
    benign_prob = benign_logit.softmax(-1)[benign_pred].item()

    axes[i, 0].imshow(np.array(benign_im))
    axes[i, 0].set_title(f'Benign: {cls_name}\nPred: {classes[benign_pred]} ({benign_prob:.2%})', fontsize=10)
    axes[i, 0].axis('off')

    # Adversarial Image
    adv_path = f'./ifgsm/{cls_name}/{cls_name}1.png'
    adv_im = Image.open(adv_path)
    adv_logit = model(transform(adv_im).unsqueeze(0).to(device))[0]
    adv_pred = adv_logit.argmax(-1).item()
    adv_prob = adv_logit.softmax(-1)[adv_pred].item()

    axes[i, 1].imshow(np.array(adv_im))
    axes[i, 1].set_title(f'Adversarial: {cls_name}\nPred: {classes[adv_pred]} ({adv_prob:.2%})', fontsize=10)
    axes[i, 1].axis('off')

# Adjust spacing for better readability
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.suptitle("Benign vs. Adversarial Image Predictions", fontsize=14, fontweight='bold')
plt.show()
