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
import csv
import os
import time
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("--input-size", type=int, default=784, help="Size of the input")
parser.add_argument("--hidden1-size", type=int, default=256, help="Size of the first hidden layer")
parser.add_argument("--hidden2-size", type=int, default=256, help="Size of the second hidden layer")
parser.add_argument("--output-size", type=int, default=10, help="Size of the output")
parser.add_argument("--free-steps", type=int, default=40, help="Number of free optimization steps")
parser.add_argument("--nudge-steps", type=int, default=5, help="Number of nudge optimization steps")
parser.add_argument("--learning-rate", type=float, default=1.0, help="Learning rate for optimization")
parser.add_argument("--beta", type=float, default=1.0, help="Beta value for weight updates")
parser.add_argument("--batch-dim", type=int, default=50, help="Batch dimension")
parser.add_argument("--n-iters", type=int, default=4000, help="Number of iterations for optimization")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--train-init", type=str, default="zeros", help="Initialization method for state during training")
parser.add_argument("--test-init", type=str, default="zeros", help="Initialization method for state during testing")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
parser.add_argument("--mr", type=float, default=0.5, help="Energy minimization rate")
parser.add_argument("--lam", type=float, default=2.0, help="Lambda value for weight updates")
parser.add_argument("--make-tsne", action="store_true", help="Make t-SNE plot")
parser.add_argument("--print-frequency", type=int, default=25, help="Frequency of printing the error")
parser.add_argument("--n-epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--output-dir", type=str, default="results", help="Output csv file name")
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
print_frequency = args.print_frequency
n_epochs = args.n_epochs
# output_file = args.output_file
train_init = args.train_init
test_init = args.test_init
test_batch_size = batch_dim

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)


if args.dataset == "mnist":
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,)),transforms.Lambda(lambda x: x.view(-1))])
    trainset = torchvision.datasets.MNIST(root='~/datasets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_dim, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='~/datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    loader = {'train':trainloader, 'test':testloader}
# Define the fixed random input x
elif args.dataset == "random":
    x = torch.randn(batch_dim,input_size)
    t = torch.tensor([i%output_size for i in range(batch_dim)],dtype=torch.long)
    target = torch.nn.functional.one_hot(t, num_classes=output_size)
else:
    raise ValueError("Invalid dataset")

# For weight updates
# lam=5.0
# beta=2.0
# beta=0.2
model = HopfieldEnergy(input_size, hidden1_size, hidden2_size, output_size, beta=beta, lam=lam)

def minimizeEnergy(model,steps,optimizer,x,h1,h2,y,target=None,print_energy=False):
    energies = []  # List to store the energy values
    for step in range(steps):
        optimizer.zero_grad()
        energy = model(x, h1, h2, y, target=target)
        energy.backward()
        optimizer.step()

        # Restrict values between 0 and 1
        h1.data = torch.clamp(h1.data, 0, 1)
        h2.data = torch.clamp(h2.data, 0, 1)
        y.data = torch.clamp(y.data, 0, 1)

        energies.append(energy.item())  # Save the energy value

    # Save copy of the internal state variables
    h1_free = h1.detach().clone()
    h2_free = h2.detach().clone()
    y_free = y.detach().clone()

    return h1_free, h2_free, y_free, energies

print("\n\n")
print("Beta: ",beta)
print("Learning rate: ",learning_rate)
print("Iterations: ",n_iters)
print("Hidden1 size: ",hidden1_size)
print("Hidden2 size: ",hidden2_size)
print("Output size: ",output_size)
print("Input size: ",input_size)
print("Batch dim: ",batch_dim)
print("Seed: ",args.seed)
print("Dataset: ",args.dataset)
print("Free phase steps: ",free_steps)
print("Nudge phase steps: ",nudge_steps)
print("Minimization rate: ",mr)
print("Lambda: ",lam)
print("\n")
mr = 0.5
# Initialize the internal state variables
batch_dim = output_size
x = torch.zeros(batch_dim,input_size)
h1 = torch.zeros(batch_dim, hidden1_size, requires_grad=True)
h2 = torch.zeros(batch_dim, hidden2_size, requires_grad=True)
y = torch.zeros(batch_dim, output_size, requires_grad=True)
optimizer = optim.SGD([h1, h2, y], lr=mr)

# h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)
# print(energies)

###############
# Training loop
###############
# x_mem = torch.zeros(n_mem,input_size) # torch.rand(n_mem,input_size)

# mask = torch.rand(n_mem, hidden1_size) < 0.2
# h1_mem = torch.rand(n_mem, hidden1_size) * mask.float()

# mask = torch.rand(n_mem, hidden2_size) < 0.2
# h2_mem = torch.rand(n_mem, hidden2_size) * mask.float()

# mask = torch.rand(n_mem, output_size) < 0.2
# y_mem = torch.rand(n_mem, output_size) * mask.float()


# x_mem = torch.zeros(n_mem,input_size)
# h1_mem = (torch.rand(n_mem, hidden1_size)<0.5).float()
# h2_mem = (torch.rand(n_mem, hidden2_size)<0.5).float()
# y_mem = (torch.rand(n_mem, output_size)<0.5).float()
n_mem = output_size
n_steps=1000
n_triggers = 1

sequences = [(i,j) for i in range(n_mem) for j in range(n_mem) if i!=j]
labels = [np.random.choice(n_mem) for i in range(len(sequences))]
print(sequences)
print(labels)
assert(0)

# Trigger inputs (should these be repreated matrices?)
x_trigger = [torch.rand(1, input_size).repeat(n_mem, 1) for i in range(n_triggers)]
# x_trigger = [torch.rand(n_mem,input_size) for i in range(n_triggers)]


# Start state for output layer
y_A_mem = torch.eye(output_size)



# Final targets
y_mem = []
for i in range(n_triggers):
    permuted_indices = torch.randperm(output_size)
    y_mem.append(torch.eye(output_size)[permuted_indices])

lr0,lr1,lr2,lr3 = 0.02,-0.02,-0.002,0.002

for itr in range(n_steps):
    w1_update, w2_update, w3_update, b1_update, b2_update, b3_update = [], [], [], [], [], []

    # Pick which input to use as the trigger
    i = itr%n_triggers
    x.data = x_trigger[i].clone().detach()
    target = y_mem[i].clone().detach()

    # Init internal state
    h1.data.uniform_(0,1)
    h2.data.uniform_(0,1)
    y.data = y_A_mem.clone().detach() # Also init output layer to fixed state

    # Clamp on target to deepen
    h1_clamp, h2_clamp, y_clamp, energies = minimizeEnergy(model,100,optimizer,x,h1,h2,target,print_energy=False)
    w1_update.append(x.t()@h1_clamp)    
    w2_update.append(h1_clamp.t()@h2_clamp)
    w3_update.append(h2_clamp.t()@y_clamp)
    b1_update.append(h1_clamp.sum(0))
    b2_update.append(h2_clamp.sum(0))
    b3_update.append(y_clamp.sum(0))

    # Allow to wander and pull up surface. (I think y should be initialized to target...)
    y.data = target.clone().detach()
    h1_free, h2_free, y_free, energies = minimizeEnergy(model,100,optimizer,x,h1,h2,y,print_energy=False)
    w1_update.append(x.t()@h1_free)
    w2_update.append(h1_free.t()@h2_free)
    w3_update.append(h2_free.t()@y_free)
    b1_update.append(h1_free.sum(0))
    b2_update.append(h2_free.sum(0))
    b3_update.append(y_free.sum(0))

    h1.data.uniform_(0,1)
    h2.data.uniform_(0,1)

    # Find where state currently settles to. Nudge towards target
    h1_free, h2_free, y_free, energies = minimizeEnergy(model,40,optimizer,x,h1,h2,y,print_energy=False)
    w1_update.append(x.t()@h1_free)
    w2_update.append(h1_free.t()@h2_free)
    w3_update.append(h2_free.t()@y_free)
    b1_update.append(h1_free.sum(0))
    b2_update.append(h2_free.sum(0))
    b3_update.append(y_free.sum(0))

    # After nudging towards target
    h1_nudge, h2_nudge, y_nudge, energies2 = minimizeEnergy(model,5,optimizer,x,h1,h2,y,target=target,print_energy=False)
    w1_update.append(x.t()@h1_nudge)
    w2_update.append(h1_nudge.t()@h2_nudge)
    w3_update.append(h2_nudge.t()@y_nudge)
    b1_update.append(h1_nudge.sum(0))
    b2_update.append(h2_nudge.sum(0))
    b3_update.append(y_nudge.sum(0))

    # Get total update
    w1_update = lr0*w1_update[0] + lr1*w1_update[1] + lr2*w1_update[2] + lr3*w1_update[3]
    w2_update = lr0*w2_update[0] + lr1*w2_update[1] + lr2*w2_update[2] + lr3*w2_update[3]
    w3_update = lr0*w3_update[0] + lr1*w3_update[1] + lr2*w3_update[2] + lr3*w3_update[3]
    b1_update = lr0*b1_update[0] + lr1*b1_update[1] + lr2*b1_update[2] + lr3*b1_update[3]
    b2_update = lr0*b2_update[0] + lr1*b2_update[1] + lr2*b2_update[2] + lr3*b2_update[3]
    b3_update = lr0*b3_update[0] + lr1*b3_update[1] + lr2*b3_update[2] + lr3*b3_update[3]

    if itr>0:
        # Update the weights
        model.w1.weight.data += w1_update
        model.w2.weight.data += w2_update
        model.w3.weight.data += w3_update
        model.b1.weight.data += b1_update
        model.b2.weight.data += b2_update
        model.b3.weight.data += b3_update

    if itr%10 == 0:

        print("Iteration: ", itr)
        print("================================================================")

        # Test whether starting from random state with trigger input leads to nearest attractor
        # Settle energy from memory state
        # x.data = torch.rand(n_mem,input_size) # This disables the trigger
        # h1.data.uniform_(0,1)
        # h2.data.uniform_(0,1)
        h1.data.uniform_(0,1)
        h2.data.uniform_(0,1)
        for i in range(n_triggers):
            x.data = x_trigger[i].clone().detach()
            h1.data.uniform_(0,1)
            h2.data.uniform_(0,1)
            y.data = y_A_mem.clone().detach() # State state
            target = y_mem[i].clone().detach()

            h1_free, h2_free, y_free, energies = minimizeEnergy(model,100,optimizer,x,h1,h2,y,print_energy=False)
            # Calculate the L2 distance between each row of state and each memory

            distances = torch.cdist(y_free, target, p=2)

            # Print the distances in a grid
            grid = []
            for i in range(distances.shape[0]):
                row = []
                for j in range(distances.shape[1]):
                    row.append(distances[i, j])
                grid.append(row)

            print(tabulate(grid, headers=[f"Memory {j+1}" for j in range(distances.shape[1])], tablefmt="grid"))










