import torch
from defaultSetup import epsilon

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

import torch
import math
from defaultSetup import epsilon

def square_attack(model, x, y, loss_fn, epsilon=epsilon, n_iters=100, p_init=0.05):
    """
    Square Attack implementation for adversarial example generation.
    
    Parameters:
    model: PyTorch model
    x: Input tensor of shape [batch_size, channels, height, width]
    y: Target labels
    loss_fn: Loss function
    epsilon: Maximum perturbation size (default from defaultSetup)
    n_iters: Number of iterations (default: 100)
    p_init: Initial probability of changing a pixel (default: 0.05)
    
    Returns:
    x_adv: Adversarial examples
    """
    def p_selection(p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get device
    device = torch.device(
        "mps" if torch.backends.mps.is_built() else
        "cuda" if torch.cuda.is_available() else
        "cpu")
    
    # Get image dimensions
    batch_size, c, h, w = x.shape
    n_features = c * h * w
    
    # Initialize adversarial examples as copies of original images
    x_adv = x.detach().clone()
    
    # Initialize with random perturbation
    # For Linf norm constraints, we use vertical stripes
    init_delta = torch.zeros_like(x_adv).to(device)
    init_delta[:, :, :, :] = torch.rand_like(init_delta[:, :, :, :]).to(device) > 0.5
    init_delta = (2 * init_delta - 1) * epsilon
    x_adv = torch.clamp(x_adv + init_delta, 0, 1)
    
    # Get initial predictions and loss
    with torch.no_grad():
        logits = model(x_adv)
        loss_init = loss_fn(logits, y)
    
    # Main attack loop
    for i in range(n_iters):
        # Probability of changing a coordinate decreases during optimization
        p = p_selection(p_init, i, n_iters)
        
        # Calculate size of perturbation squares
        s = max(int(round(math.sqrt(p * n_features / c))), 1)
        s = min(s, h-1)  # Ensure square size is valid
        
        # Create perturbations batch-wise
        x_new = x_adv.clone()
        
        for idx in range(batch_size):
            # Choose random center for the perturbation square
            center_h = torch.randint(0, h - s + 1, (1,), device=device).item()
            center_w = torch.randint(0, w - s + 1, (1,), device=device).item()
            
            # Create perturbation for this instance
            # Generate random noise of -eps or +eps
            noise = (2 * (torch.rand(c, s, s, device=device) > 0.5).float() - 1) * epsilon
            noise = noise.to(device)
            
            # Apply perturbation to the square region
            new_x_window = x[idx, :, center_h:center_h+s, center_w:center_w+s] + noise
            new_x_window = torch.clamp(new_x_window, 0, 1)
            
            # Only apply perturbation if it creates a different image
            if torch.sum(torch.abs(new_x_window - x_adv[idx, :, center_h:center_h+s, center_w:center_w+s])) > 1e-7:
                x_new[idx, :, center_h:center_h+s, center_w:center_w+s] = new_x_window
        
        # Evaluate new candidates
        with torch.no_grad():
            logits_new = model(x_new)
            loss_new = loss_fn(logits_new, y)
        
        # Update adversarial examples when loss increases
        # For untargeted attacks, we want to maximize the loss
        improved = loss_new > loss_init
        x_adv[improved] = x_new[improved]
        loss_init[improved] = loss_new[improved]
        
        # Ensure we stay within epsilon constraint (L-inf norm)
        delta = x_adv - x
        delta = torch.clamp(delta, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
    
    return x_adv

if __name__ == "__main__":
    print("Square attack function ready for adversarial examples generation.")

if __name__ == "__main__":
    print("Adversarial attack functions ready.")
