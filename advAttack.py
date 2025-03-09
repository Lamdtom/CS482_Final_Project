import torch
import torch.nn.functional as F
from defaultSetup import epsilon, alpha

def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y) # calculate loss
    loss.backward() # calculate gradient
    # fgsm: use gradient ascent on x_adv to maximize loss
    x_adv = x_adv + epsilon * x_adv.grad.detach().sign()
    return x_adv

def pgd(model, x, y, loss_fn, epsilon=8/255, alpha=2/255, num_iter=50):
    # Initialize x_adv as the original benign image x
    
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)  # Random start
    x_adv = torch.clamp(x_adv, 0, 1)  # Ensure it's a valid image
    x_adv.requires_grad = True
    # Iterative attack
    for i in range(num_iter):
        # Use FGSM with step size alpha for the update
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)

        # Project adversarial example back onto ε-ball around x
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)  # Projection step
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid pixel range

    return x_adv



def ifgsm(model, x, y, loss_fn, epsilon=0.1, alpha=0.005, num_iter=20, lambda_reg=0.01):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    # write a loop of num_iter to represent the iterative times
    # for each loop
    for i in range(num_iter):
        # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
        # clip new x_adv back to [x-epsilon, x+epsilon]
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

        # Minimize L2 distance between x_adv and the original input x
        distance = torch.norm(x_adv - x, p=2)  # Compute L2 distance between x_adv and x
        # Add regularization term to penalize large distances
        x_adv = x_adv - lambda_reg * (x_adv - x) / distance  # Adjust the adversarial example
    return x_adv

if __name__ == "__main__":
    print("Adversarial attack functions ready.")
