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
    x_adv.requires_grad = True

    # Loop for num_iter iterations
    for _ in range(num_iter):
        # Compute the gradient using FGSM with step size alpha
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
        # Project the perturbed image back to the epsilon-ball around x
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv


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
