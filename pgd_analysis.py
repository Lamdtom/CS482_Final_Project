# pgd_analysis.py
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
from defaultSetup import transform, device
from main import load_model

# Define class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Select model
selected_model = 'efficientnet_b0'
checkpoint_path = f"./checkpoints/{selected_model}_checkpoint.pth"
model = load_model(selected_model, checkpoint_path=checkpoint_path, num_classes=10)
model.eval()

# Create a figure with 3 columns (Benign | PGD | Difference)
fig, axes = plt.subplots(len(classes), 3, figsize=(12, 20))

# Track statistics
total_fooled = 0
fooled_classes = []

for i, cls_name in enumerate(classes):
    # Benign Image
    benign_path = f'./data/{cls_name}/{cls_name}1.png'
    benign_im = Image.open(benign_path)
    benign_array = np.array(benign_im)
    benign_logit = model(transform(benign_im).unsqueeze(0).to(device))[0]
    benign_pred = benign_logit.argmax(-1).item()
    benign_prob = benign_logit.softmax(-1)[benign_pred].item()

    axes[i, 0].imshow(benign_array)
    axes[i, 0].set_title(f'Original: {cls_name}\nPred: {classes[benign_pred]} ({benign_prob:.2%})', 
                        fontsize=9)
    axes[i, 0].axis('off')

    # PGD Image
    pgd_path = f'./pgd/{cls_name}/{cls_name}1.png'
    
    if os.path.exists(pgd_path):
        pgd_im = Image.open(pgd_path)
        pgd_array = np.array(pgd_im)
        pgd_logit = model(transform(pgd_im).unsqueeze(0).to(device))[0]
        pgd_pred = pgd_logit.argmax(-1).item()
        pgd_prob = pgd_logit.softmax(-1)[pgd_pred].item()
        
        axes[i, 1].imshow(pgd_array)
        
        # Check if prediction changed
        if pgd_pred != benign_pred:
            title_color = 'red'
            total_fooled += 1
            fooled_classes.append(cls_name)
            title = f'PGD: {cls_name}\nFOOLED: {classes[pgd_pred]} ({pgd_prob:.2%})'
        else:
            title_color = 'black'
            title = f'PGD: {cls_name}\nPred: {classes[pgd_pred]} ({pgd_prob:.2%})'
            
        axes[i, 1].set_title(title, fontsize=9, color=title_color)
        
        # Visualize perturbation (difference between original and adversarial)
        diff = np.abs(pgd_array.astype(np.float32) - benign_array.astype(np.float32))
        
        # Amplify difference for better visibility (optional)
        diff = np.clip(diff * 10, 0, 255).astype(np.uint8)
        
        axes[i, 2].imshow(diff)
        axes[i, 2].set_title(f'Perturbation\n(Amplified 10x)', fontsize=9)
    else:
        axes[i, 1].text(0.5, 0.5, "PGD image not found", ha='center', va='center')
        axes[i, 2].text(0.5, 0.5, "No difference", ha='center', va='center')
    
    axes[i, 1].axis('off')
    axes[i, 2].axis('off')

# Add summary information
plt.figtext(0.5, 0.01, f"Attack Success Rate: {total_fooled}/{len(classes)} classes ({total_fooled/len(classes):.0%})\n"
             f"Fooled Classes: {', '.join(fooled_classes)}", 
             ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.suptitle(f"PGD Attack Analysis - Model: {selected_model}", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()