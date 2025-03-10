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

def pgd(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    # Initialize x_adv as the original benign image x
    x_adv = x.detach().clone()
    
    # Add random initialization within epsilon ball
    for i in range(x_adv.shape[1]):  # For each channel
        eps_value = epsilon[i].item()
        # Generate uniform random noise within [-epsilon, epsilon]
        noise = torch.zeros_like(x_adv[:, i, :, :]).uniform_(-eps_value, eps_value)
        # Add noise to the image
        x_adv[:, i, :, :] += noise
    
    # Ensure valid image range after adding noise
    x_adv = torch.clamp(x_adv, 0, 1)
    x_adv.requires_grad = True

    # Loop for num_iter iterations (rest of the function remains the same)
    for _ in range(num_iter):
        # Compute the gradient using FGSM with step size alpha
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
        # Project the perturbed image back to the epsilon-ball around x
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv


def pgd_with_restarts(model, x, y, loss_fn, epsilon=epsilon*0.6, alpha=alpha, num_iter=20, num_restarts=5):
    best_loss = None
    best_x_adv = None
    
    for _ in range(num_restarts):
        # Run standard PGD
        x_adv = pgd(model, x, y, loss_fn, epsilon, alpha, num_iter)
        
        # Evaluate this result
        with torch.no_grad():
            loss = loss_fn(model(x_adv), y)
        
        # Keep track of the best result
        if best_loss is None or loss > best_loss:
            best_loss = loss
            best_x_adv = x_adv.clone()
    
    return best_x_adv


def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    # initialize x_adv as original benign image x
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    # write a loop of num_iter to represent the iterative times
    # for each loop
    for i in range(num_iter):
        # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
        # clip new x_adv back to [x-epsilon, x+epsilon]
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv

if __name__ == "__main__":
    print("Adversarial attack functions ready.")
