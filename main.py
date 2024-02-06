import torch
import torch.nn as nn
import torch.optim as optim
from utils import HopfieldEnergy, HopfieldUpdate
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision
import torchvision.transforms as transforms
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-size", type=int, default=50, help="Size of the input")
parser.add_argument("--hidden1-size", type=int, default=100, help="Size of the first hidden layer")
parser.add_argument("--hidden2-size", type=int, default=100, help="Size of the second hidden layer")
parser.add_argument("--output-size", type=int, default=15, help="Size of the output")
parser.add_argument("--free-steps", type=int, default=40, help="Number of free optimization steps")
parser.add_argument("--nudge-steps", type=int, default=5, help="Number of nudge optimization steps")
parser.add_argument("--learning-rate", type=float, default=1.0, help="Learning rate for optimization")
parser.add_argument("--beta", type=float, default=1.0, help="Beta value for weight updates")
parser.add_argument("--batch-dim", type=int, default=15, help="Batch dimension")
parser.add_argument("--n-iters", type=int, default=3000, help="Number of iterations for optimization")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--init", type=str, default="random", help="Initialization method for weights")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
parser.add_argument("--mr", type=float, default=0.1, help="Energy minimization rate")
parser.add_argument("--lam", type=float, default=1.0, help="Lambda value for weight updates")
args = parser.parse_args()


input_size = args.input_size
hidden1_size = args.hidden1_size
hidden2_size = args.hidden2_size
output_size = args.output_size
free_steps = args.free_steps
nudge_steps = args.nudge_steps
learning_rate = args.learning_rate # 1 - 200
beta = args.beta # 1-40
batch_dim = args.batch_dim
n_iters = args.n_iters
mr = args.mr #0.1
lam = args.lam

if args.dataset == "mnist":
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,)),transforms.Lambda(lambda x: x.view(-1))])
    trainset = torchvision.datasets.MNIST(root='~/datasets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_dim, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='~/datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_dim, shuffle=False, num_workers=2)


# Define the fixed random input x
if args.dataset == "random":
    x = torch.randn(batch_dim,input_size)
    t = torch.tensor([i%output_size for i in range(batch_dim)],dtype=torch.long)
elif args.dataset == "mnist":
    for batch in trainloader: # Get one MNIST batch
        x,t = batch
        break
else:
    raise ValueError("Invalid dataset")

target = torch.nn.functional.one_hot(t, num_classes=output_size)



# Define the optimizer

# For weight updates
model = HopfieldEnergy(input_size, hidden1_size, hidden2_size, output_size, beta=beta, lam=lam)


# Optimization loop
print("\n\n")
print(r"$\beta$: ",beta)
print("Learning rate: ",learning_rate)
print("Iterations: ",n_iters)
print("Free phase steps: ",free_steps)
print("Nudge phase steps: ",nudge_steps)
print("Minimization rate: ",mr)
print("batch_dim: ",batch_dim)
print("\n\n")

error = []

###############
# Training loop
###############

for itr in range(n_iters): # Random

# for itr,batch in enumerate(trainloader): # MNIST
#     x = batch[0]
#     target = torch.nn.functional.one_hot(batch[1], num_classes=10)

    energies = []  # List to store the energy values

    # Initialize internal state variables
    if args.init == "zeros":
        h1 = torch.zeros(batch_dim, hidden1_size, requires_grad=True)
        h2 = torch.zeros(batch_dim, hidden2_size, requires_grad=True)
        y = torch.zeros(batch_dim, output_size, requires_grad=True)
    elif args.init == "random":
        h1 = torch.rand(batch_dim, hidden1_size, requires_grad=True)
        h2 = torch.rand(batch_dim, hidden2_size, requires_grad=True)
        y = torch.rand(batch_dim, output_size, requires_grad=True)
    else:
        raise ValueError("Invalid initialization method")

    optimizer = optim.SGD([h1, h2, y], lr=mr)

    for step in range(free_steps):
        optimizer.zero_grad()
        energy = model(x, h1, h2, y)
        energy.backward()
        optimizer.step()

        # Restrict values between 0 and 1
        h1.data = torch.clamp(h1.data, 0, 1)
        h2.data = torch.clamp(h2.data, 0, 1)
        y.data = torch.clamp(y.data, 0, 1)

        energies.append(energy.item())  # Save the energy value
        # print(f'Step {step}, Energy: {energy.item()}')

    # Save copy of the internal state variables
    h1_free = h1.detach().clone()
    h2_free = h2.detach().clone()
    y_free = y.detach().clone()


    if (itr+1)%(n_iters//10) == 0:
        print("Iteration: ",itr+1)
        # print("Output: ",y_free)
        print("Error: ",(y_free-target).pow(2).sum())
        prediction = torch.argmax(y_free, dim=1)
        accuracy = torch.mean((prediction==t).float())
        print("Accuracy: ",accuracy)
        if accuracy < 0.2:
            assert(0)   # Kill if not learning

    error.append((y_free-target).pow(2).sum())

    for step in range(nudge_steps):
        optimizer.zero_grad()
        energy = model(x, h1, h2, y, target=target)
        energy.backward()
        optimizer.step()

        # Restrict values between 0 and 1
        h1.data = torch.clamp(h1.data, 0, 1)
        h2.data = torch.clamp(h2.data, 0, 1)
        y.data = torch.clamp(y.data, 0, 1)

        energies.append(energy.item())  # Save the energy value
        # print(f'Step {step}, Energy: {energy.item()}')

    # Save copy of the internal state variables
    h1_nudge = h1.detach().clone()
    h2_nudge = h2.detach().clone()
    y_nudge = y.detach().clone()

    # Calculate the weight updates
    w1_update = (x.t() @ h1_nudge - x.t() @ h1_free)/(beta*batch_dim)
    w2_update = (h1_nudge.t() @ h2_nudge - h1_free.t() @ h2_free)/(beta*batch_dim)
    w3_update = (h2_nudge.t() @ y_nudge - h2_free.t() @ y_free)/(beta*batch_dim)
    # Correct?
    b1_update = (h1_nudge - h1_free).sum(0)/(beta*batch_dim)
    b2_update = (h2_nudge - h2_free).sum(0)/(beta*batch_dim)
    b3_update = (y_nudge - y_free).sum(0)/(beta*batch_dim)

    # Update the weights
    model.w1.weight.data += learning_rate * w1_update
    model.w2.weight.data += learning_rate * w2_update
    model.w3.weight.data += learning_rate * w3_update
    model.b1.weight.data += learning_rate * b1_update
    model.b2.weight.data += learning_rate * b2_update
    model.b3.weight.data += learning_rate * b3_update

    if (itr+1)%(n_iters//20) == 0:
        # Plot the energies
        plt.plot(energies,label=str(itr))
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.title('Energy vs. Step')
        plt.legend()
        plt.savefig('energy_vs_step.png')
    if (itr+1)%(n_iters//200) == 0:
        # Plot the error
        plt.figure()
        plt.plot(error)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Error vs. Iteration\nbeta: '+str(beta)+'\nlearning_rate: '+str(learning_rate)+'\nError: '+str(error[-1]))
        plt.savefig('error_vs_iteration_beta_'+str(beta)+'_lr_'+str(learning_rate)+'.png')
        plt.close()
            
######################################################################################################
# Generate random batch of points
n_samples = 100

x_test = x.repeat(n_samples,1).clone()
# target_test = target.repeat(n_samples,1)

h1 = torch.rand(batch_dim*n_samples, hidden1_size, requires_grad=True)
h2 = torch.rand(batch_dim*n_samples, hidden2_size, requires_grad=True)
y = torch.rand(batch_dim*n_samples, output_size, requires_grad=True)

optimizer = optim.SGD([h1, h2, y], lr=mr)

for step in range(free_steps):
    optimizer.zero_grad()
    energy = model(x_test, h1, h2, y)
    energy.backward()
    optimizer.step()

    # Restrict values between 0 and 1
    h1.data = torch.clamp(h1.data, 0, 1)
    h2.data = torch.clamp(h2.data, 0, 1)
    y.data = torch.clamp(y.data, 0, 1)

    energies.append(energy.item())  # Save the energy value
    # print(f'Step {step}, Energy: {energy.item()}')

# Save copy of the internal state variables
h1_free = h1.detach().clone()
h2_free = h2.detach().clone()
y_free = y.detach().clone()

# print(y_free)

# # t-SNE plot to visualize clusters in h1_free
# from sklearn.manifold import TSNE

# colors = [t[i % batch_dim] for i in range(batch_dim * n_samples)]  # Example color vector
# cmap = cm.get_cmap('viridis')
# s, alpha = 2, 0.02

# X_embedded_h1 = TSNE(n_components=2).fit_transform(h1_free.numpy())
# plt.subplot(1, 3, 1)
# plt.scatter(X_embedded_h1[:, 0], X_embedded_h1[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
# plt.title('hidden 1')
# plt.gca().set_aspect('equal')

# # t-SNE plot to visualize clusters in h2_free
# X_embedded_h2 = TSNE(n_components=2).fit_transform(h2_free.numpy())
# plt.subplot(1, 3, 2)
# plt.scatter(X_embedded_h2[:, 0], X_embedded_h2[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
# plt.title('hidden 2')
# plt.gca().set_aspect('equal')

# # t-SNE plot to visualize clusters in y_free
# X_embedded_y = TSNE(n_components=2).fit_transform(y_free.numpy())
# plt.subplot(1, 3, 3)
# plt.scatter(X_embedded_y[:, 0], X_embedded_y[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
# plt.title('output')
# plt.gca().set_aspect('equal')

# plt.tight_layout()
# plt.figure(figsize=(10, 6))  # Adjust the figure size
# plt.savefig('clusters.png')
