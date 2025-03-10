import matplotlib.pyplot as plt
from PIL import Image
from defaultSetup import transform, device
import numpy as np
from main import load_model

# Define class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Select the model you want to load
selected_model = 'efficientnet_b0'
checkpoint_path = f"./checkpoints/{selected_model}_checkpoint.pth"
# Load the model
model = load_model(selected_model, checkpoint_path=checkpoint_path, num_classes=10)
model.eval()

# Create a figure with 2 columns (Benign | PGD)
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

    # PGD Adversarial Image
    pgd_path = f'./pgd/{cls_name}/{cls_name}1.png'
    pgd_im = Image.open(pgd_path)
    pgd_logit = model(transform(pgd_im).unsqueeze(0).to(device))[0]
    pgd_pred = pgd_logit.argmax(-1).item()
    pgd_prob = pgd_logit.softmax(-1)[pgd_pred].item()

    axes[i, 1].imshow(np.array(pgd_im))
    
    # Color the title red if prediction is wrong (model fooled)
    if pgd_pred != benign_pred:
        title_color = 'red'
    else:
        title_color = 'black'
        
    axes[i, 1].set_title(f'PGD: {cls_name}\nPred: {classes[pgd_pred]} ({pgd_prob:.2%})', 
                         fontsize=10, color=title_color)
    axes[i, 1].axis('off')

# Adjust spacing for better readability
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.suptitle("Benign vs. PGD Adversarial Image Predictions", fontsize=14, fontweight='bold')
plt.show()