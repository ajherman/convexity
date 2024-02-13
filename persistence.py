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
batch_dim = 10
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
n_mem = 10
x_mem = torch.zeros(n_mem,input_size) # torch.rand(n_mem,input_size)

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

n_steps=200
x_trigger = torch.rand(n_mem,input_size) # Input that will cause multi-state attractor

x_mem = torch.randn(n_mem,input_size) # Memories to store
y_mem = torch.eye(output_size)
for itr in range(n_steps):

    # h1.data.uniform_(0,1)
    # h2.data.uniform_(0,1)
    # y.data.uniform_(0,1)

    # # Settle energy from random state
    # free_steps = 100
    # h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)
    # # Get l2 distance between free states and memory states. Try all layers as one tensor
    # l2 = (h1_free-h1_mem).pow(2).sum(dim=1) + (h2_free-h2_mem).pow(2).sum(dim=1) + (y_free-y_mem).pow(2).sum(dim=1)
    # # l2 = (h1_free-h1_mem).pow(2).sum() + (h2_free-h2_mem).pow(2).sum() + (y_free-y_mem).pow(2).sum()
    # min_index = torch.argmin(l2)
    # print("L2 distance: ", torch.min(l2))
    # print("Index of min element: ", min_index)

    # # Update the weights
    # alpha=20.0
    # model.w1.weight.data += learning_rate * (w1_update - alpha*x.t() @ h1_free)
    # model.w2.weight.data += learning_rate * (w2_update - alpha*h1_free.t() @ h2_free)
    # model.w3.weight.data += learning_rate * (w3_update - alpha*h2_free.t() @ y_free)
    # model.b1.weight.data += learning_rate * (b1_update - alpha*h1_free.sum(0))
    # model.b2.weight.data += learning_rate * (b2_update - alpha*h2_free.sum(0))
    # model.b3.weight.data += learning_rate * (b3_update - alpha*y_free.sum(0))


# #########################################################

#     x.data = x_mem.data.clone().detach()
#     h1.data = h1_mem.data.clone().detach()
#     h2.data = h2_mem.data.clone().detach()
#     y.data = y_mem.data.clone().detach()


#     # Settle energy from memory state
#     free_steps = 200
#     h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)

#     # Print L2 distances
#     l2 = (h1_free-h1_mem).pow(2).sum(dim=1) + (h2_free-h2_mem).pow(2).sum(dim=1) + (y_free-y_mem).pow(2).sum(dim=1)
#     print("L2 distance: ", l2)
#     print("Index of min element: ", torch.argmin(l2))

#     # Update the weights
#     alpha=1.0
#     model.w1.weight.data += learning_rate * (x_mem.t()@h1_mem - alpha*x_mem.t() @ h1_free)
#     model.w2.weight.data += learning_rate * (h1_mem.t()@h2_mem - alpha*h1_free.t() @ h2_free)
#     model.w3.weight.data += learning_rate * (h2_mem.t()@y_mem - alpha*h2_free.t() @ y_free)
#     model.b1.weight.data += learning_rate * (h1_mem - alpha*h1_free).sum(0)
#     model.b2.weight.data += learning_rate * (h2_mem - alpha*h2_free).sum(0)
#     model.b3.weight.data += learning_rate * (y_mem - alpha*y_free).sum(0)

#     if _%10 == 0:
#         x.data = x_mem.data.clone().detach()#+0.1*torch.randn_like(x_mem)
#         h1.data = h1_mem.data.clone().detach()+0.5*torch.randn_like(h1_mem)
#         h2.data = h2_mem.data.clone().detach()+0.5*torch.randn_like(h2_mem)
#         y.data = y_mem.data.clone().detach()+0.5*torch.randn_like(y_mem)

#         h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)

#         # Calculate the L2 distance between each row of state and each memory
#         distances = torch.cdist(torch.cat([h1_free, h2_free, y_free],dim=1), torch.cat([h1_mem, h2_mem, y_mem],dim=1), p=2)

#         # Print the distances in a grid
#         grid = []
#         for i in range(distances.shape[0]):
#             row = []
#             for j in range(distances.shape[1]):
#                 row.append(distances[i, j])
#             grid.append(row)

#         print(tabulate(grid, headers=[f"Memory {j+1}" for j in range(distances.shape[1])], tablefmt="grid"))



#########################################################

    lr1,lr2 = 0.05,0.05

    x.data = x_trigger.data.clone().detach()
    h1.data.uniform_(0,1)
    h2.data.uniform_(0,1)
    target = y_mem.data.clone().detach()


    # Settle energy from memory state
    free_steps = 100
    
    #############################
    # Create multistate attractor
    #############################

    # Find the state we want to deepen (target clamped)
    h1_clamp, h2_clamp, y_clamp, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,target,print_energy=False)

    # Find contrast state
    h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)

    if itr%10 == 0:
        # # Print L2 distances
        l2 = 0.0*(h1_free-h1_clamp).pow(2).sum(dim=1) +0.0*(h2_free-h2_clamp).pow(2).sum(dim=1) +(y_free-y_clamp).pow(2).sum(dim=1)
        print("L2 distance: ", l2)
        print("Index of min element: ", torch.argmin(l2))

    # Update the weights
    alpha=1.0
    learning_rate = lr1 #0.05
    model.w1.weight.data += learning_rate * (x_mem.t()@h1_clamp - alpha*x_mem.t() @ h1_free)
    model.w2.weight.data += learning_rate * (h1_clamp.t()@h2_clamp - alpha*h1_free.t() @ h2_free)
    model.w3.weight.data += learning_rate * (h2_clamp.t()@y_mem - alpha*h2_free.t() @ y_free)
    model.b1.weight.data += learning_rate * (h1_clamp - alpha*h1_free).sum(0)
    model.b2.weight.data += learning_rate * (h2_clamp - alpha*h2_free).sum(0)
    model.b3.weight.data += learning_rate * (y_clamp - alpha*y_free).sum(0)

    ##################
    # Regular training
    ##################

    # Settle energy from memory state
    x.data = x_mem.data.clone().detach()
    h1.data.uniform_(0,1)
    h2.data.uniform_(0,1)
    y.data.uniform_(0,1)

    free_steps = 40
    h1_free, h2_free, y_free, energies1 = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)

    if itr%10 == 0:
        # print(y_free)
        l2 = (y_free-target).pow(2).sum(dim=1)
        print("Learning rate: ",learning_rate)
        print("beta: ",beta)
        print("L2 distance: ", l2)
        print("Index of min element: ", torch.argmin(l2))

    nudge_steps = 5
    h1_nudge, h2_nudge, y_nudge, energies2 = minimizeEnergy(model,nudge_steps,optimizer,x,h1,h2,y,target=target,print_energy=False)

    # Calculate the weight updates
    w1_update = (x.t() @ h1_nudge - x.t() @ h1_free)/(beta*batch_dim)
    w2_update = (h1_nudge.t() @ h2_nudge - h1_free.t() @ h2_free)/(beta*batch_dim)
    w3_update = (h2_nudge.t() @ y_nudge - h2_free.t() @ y_free)/(beta*batch_dim)
    b1_update = (h1_nudge - h1_free).sum(0)/(beta*batch_dim)
    b2_update = (h2_nudge - h2_free).sum(0)/(beta*batch_dim)
    b3_update = (y_nudge - y_free).sum(0)/(beta*batch_dim)
    
    # Update the weights
    learning_rate = lr2
    model.w1.weight.data += learning_rate * w1_update
    model.w2.weight.data += learning_rate * w2_update
    model.w3.weight.data += learning_rate * w3_update
    model.b1.weight.data += learning_rate * b1_update
    model.b2.weight.data += learning_rate * b2_update
    model.b3.weight.data += learning_rate * b3_update


    if itr%20 == 0:

        # Test whether starting from random state with trigger input leads to nearest attractor
        # Settle energy from memory state
        x.data = x_mem.data.clone().detach()
        h1.data.uniform_(0,1)
        h2.data.uniform_(0,1)
        y.data.uniform_(0,1)
        free_steps = 100
        h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)
        free_steps = 50
        h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x_trigger,h1,h2,y,print_energy=False)
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
















# sample_errors = []
# for epoch in range(n_epochs):
#     print("Epoch: ",epoch)
#     for itr,batch in enumerate(trainloader): # MNIST
#         if itr%2==0:
#             x1,t1 = batch
#             continue
#         else:
#             x2,t2 = batch

#         # t = (t1+t2).remainder(10)
#         t = t1
#         target = torch.nn.functional.one_hot(t, num_classes=10)

#         # Initialize internal state variables
#         if args.train_init == "zeros":
#             h1.data.zero_()
#             h2.data.zero_()
#             y.data.zero_()
#         elif args.train_init == "random":
#             h1.data.uniform_(0,1)
#             h2.data.uniform_(0,1)
#             y.data.uniform_(0,1)
#         elif args.train_init == "previous":
#             pass
#         else:
#             raise ValueError("Invalid initialization method")

#         # Free phase        
#         h1_free, h2_free, y_free, phase1_energies = minimizeEnergy(model,free_steps,optimizer,x1,h1,h2,y,print_energy=False)

#         if (itr+1)%(n_iters//print_frequency) != 0: # Skip when testing
#             h1_nudge, h2_nudge, y_nudge, phase2_energies = minimizeEnergy(model,nudge_steps,optimizer,x1,h1,h2,y,target=target,print_energy=False)
#             # Calculate the weight updates
#             w1_update = (x1.t() @ h1_nudge - x1.t() @ h1_free)/(beta*batch_dim)
#             w2_update = (h1_nudge.t() @ h2_nudge - h1_free.t() @ h2_free)/(beta*batch_dim)
#             w3_update = (h2_nudge.t() @ y_nudge - h2_free.t() @ y_free)/(beta*batch_dim)
#             b1_update = (h1_nudge - h1_free).sum(0)/(beta*batch_dim)
#             b2_update = (h2_nudge - h2_free).sum(0)/(beta*batch_dim)
#             b3_update = (y_nudge - y_free).sum(0)/(beta*batch_dim)

#             # Update the weights
#             model.w1.weight.data += learning_rate * w1_update
#             model.w2.weight.data += learning_rate * w2_update
#             model.w3.weight.data += learning_rate * w3_update
#             model.b1.weight.data += learning_rate * b1_update
#             model.b2.weight.data += learning_rate * b2_update
#             model.b3.weight.data += learning_rate * b3_update
#         else:
#             print("Check if x1 is learning t1...")
#             sample_error = (y_free-target).pow(2).sum(dim=1).mean()
#             sample_errors.append(sample_error.item())
#             print("MSE: ",sample_error.item())
#             prediction = torch.argmax(y_free, dim=1)
#             accuracy = torch.mean((prediction==t).float())
#             # if accuracy < 0.5:
#             #     assert(0)   # Kill if not learning
#             print("Accuracy: ",accuracy.item())

#         # Starting from previous state (either after nudge or free phase)
#         if (itr+1)%(n_iters//print_frequency) != 0:
#             h1_nudge, h2_nudge, y_nudge, phase4_energies = minimizeEnergy(model,nudge_steps,optimizer,x2,h1,h2,y,target=target,print_energy=False)
#             # Calculate the weight updates
#             w1_update = (x2.t() @ h1_nudge - 0*x2.t() @ h1_free)/(beta*batch_dim)
#             w2_update = (h1_nudge.t() @ h2_nudge - 0*h1_free.t() @ h2_free)/(beta*batch_dim)
#             w3_update = (h2_nudge.t() @ y_nudge - 0*h2_free.t() @ y_free)/(beta*batch_dim)
#             b1_update = (h1_nudge - 0*h1_free).sum(0)/(beta*batch_dim)
#             b2_update = (h2_nudge - 0*h2_free).sum(0)/(beta*batch_dim)
#             b3_update = (y_nudge - 0*y_free).sum(0)/(beta*batch_dim)

#             # Update the weights
#             model.w1.weight.data += learning_rate * w1_update
#             model.w2.weight.data += learning_rate * w2_update
#             model.w3.weight.data += learning_rate * w3_update
#             model.b1.weight.data += learning_rate * b1_update
#             model.b2.weight.data += learning_rate * b2_update
#             model.b3.weight.data += learning_rate * b3_update
            
#         else:
#             print("Check actual learning on x2...")
#             h1_free, h2_free, y_free, phase3_energies = minimizeEnergy(model,free_steps,optimizer,x2,h1,h2,y,print_energy=False)
#             print("\nIteration: ",itr+1)
#             sample_error = (y_free-target).pow(2).sum(dim=1).mean()
#             sample_errors.append(sample_error.item())
#             print("MSE: ",sample_error.item())
#             prediction = torch.argmax(y_free, dim=1)
#             accuracy = torch.mean((prediction==t).float())
#             # if accuracy < 0.5:
#             #     assert(0)   # Kill if not learning
#             print("Accuracy: ",accuracy.item())

        # Nudge phase
        # h1_nudge, h2_nudge, y_nudge, phase4_energies = minimizeEnergy(model,nudge_steps,optimizer,x2,h1,h2,y,target=target,print_energy=False)
        # energies = phase1_energies + phase2_energies + phase3_energies + phase4_energies
        
        # # Calculate the weight updates
        # w1_update = (x2.t() @ h1_nudge - x2.t() @ h1_free)/(beta*batch_dim)
        # w2_update = (h1_nudge.t() @ h2_nudge - h1_free.t() @ h2_free)/(beta*batch_dim)
        # w3_update = (h2_nudge.t() @ y_nudge - h2_free.t() @ y_free)/(beta*batch_dim)
        # b1_update = (h1_nudge - h1_free).sum(0)/(beta*batch_dim)
        # b2_update = (h2_nudge - h2_free).sum(0)/(beta*batch_dim)
        # b3_update = (y_nudge - y_free).sum(0)/(beta*batch_dim)

        # # Update the weights
        # model.w1.weight.data += learning_rate * w1_update
        # model.w2.weight.data += learning_rate * w2_update
        # model.w3.weight.data += learning_rate * w3_update
        # model.b1.weight.data += learning_rate * b1_update
        # model.b2.weight.data += learning_rate * b2_update
        # model.b3.weight.data += learning_rate * b3_update

        # if (itr+1)%200 == 0:
        #     # Plot the energies
        #     plt.plot(energies,label=str(itr))
        #     plt.xlabel('Step')
        #     plt.ylabel('Energy')
        #     plt.title('Energy vs. Step')
        #     plt.legend()
        #     plt.savefig('energy_vs_step.png')

        # if (itr+1)%(n_iters//print_frequency) == 0:
        #     # Plot the error
        #     plt.figure()
        #     plt.plot(sample_errors)
        #     plt.xlabel('Iteration')
        #     plt.ylabel('MSE')
        #     plt.title('MSE vs. Iteration\nbeta: '+str(beta)+'\nlearning_rate: '+str(learning_rate)+'\nMSE: '+str(sample_errors[-1]))
        #     plt.savefig('error_vs_iteration_beta_'+str(beta)+'_lr_'+str(learning_rate)+'.png')
        #     plt.close()

#     # Testing (regular)
#     ######################################################################################################
#     errors, accuracies = {'train':[],'test':[]}, {'train':[],'test':[]}
#     train_x, train_h1, train_h2, train_y, train_t, train_target = [], [], [], [], [], []
#     for split in ['train','test']:
#         for itr,batch in enumerate(loader[split]):
#             x,t = batch
#             target = torch.nn.functional.one_hot(t, num_classes=10)
#             if args.test_init == "zeros":
#                 h1.data.zero_()
#                 h2.data.zero_()
#                 y.data.zero_()
#             elif args.test_init == "random":
#                 h1.data.uniform_(0,1)
#                 h2.data.uniform_(0,1)
#                 y.data.uniform_(0,1)
#             elif args.test_init == "previous":
#                 pass
#             else:
#                 raise ValueError("Invalid initialization method")
#             h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)
            
#             # Save internal state variables
#             if split == 'train':
#                 train_x.append(x)
#                 train_h1.append(h1_free)
#                 train_h2.append(h2_free)
#                 train_y.append(y_free)
#                 train_t.append(t)
#                 train_target.append(target)

#             error = (y_free-target).pow(2).sum(dim=1).mean()
#             prediction = torch.argmax(y_free, dim=1)
#             accuracy = torch.mean((prediction==t).float())
#             errors[split].append(error.item())
#             accuracies[split].append(accuracy.item())
#         mean_error = np.mean(errors[split])
#         mean_accuracy = np.mean(accuracies[split])
#         print("Split: ",split)
#         print("Accuracy: ",mean_accuracy)
#         print("MSE: ",mean_error)

#     # Save internal state variables
#     train_x = torch.cat(train_x)
#     train_h1 = torch.cat(train_h1)
#     train_h2 = torch.cat(train_h2)
#     train_y = torch.cat(train_y)
#     train_t = torch.cat(train_t)
#     train_target = torch.cat(train_target)

#     # Write test error to CSV file
#     with open(args.output_dir+'/results.csv', 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([epoch,np.mean(errors['train']),np.mean(accuracies['train']),np.mean(errors['test']),np.mean(accuracies['test'])])

# # ###############################################################################################
# ##################
# # Repeater testing
# ##################
#     tic = time.time()
#     # Get a small batch
#     if args.dataset == "mnist":
#         idx = np.random.randint(1000)
#         x_test,t_test = train_x[idx:idx+test_batch_size], train_t[idx:idx+test_batch_size]
#     # elif args.dataset == "random":
#     #     x_test, t_test, target_test = x, t, target
#     else:
#         raise ValueError("Invalid dataset")
    
#     # Generate random batch of points
#     n_samples = 1000

#     # Expand dataset and internal state variables
#     permutation = True
#     if permutation: # Get permuted settled states
#         perm = torch.randperm(test_batch_size * n_samples)
#         x_test = train_x[idx:idx+test_batch_size].repeat(n_samples,1).clone()
#         h1_test = train_h1[idx:idx+test_batch_size].repeat(n_samples, 1)[perm].clone().requires_grad_(True)
#         h2_test = train_h2[idx:idx+test_batch_size].repeat(n_samples, 1)[perm].clone().requires_grad_(True)
#         y_test = train_y[idx:idx+test_batch_size].repeat(n_samples, 1)[perm].clone().requires_grad_(True)
#         target_test = train_target[idx:idx+test_batch_size].repeat(n_samples, 1)[perm].clone()
#         t_test = train_t[idx:idx+test_batch_size].repeat(n_samples).clone()
#     else: # Original version with randomly initialize internal state
#         x_test = train_x[idx:idx+test_batch_size].repeat(n_samples,1).clone()
#         h1_test = torch.rand(test_batch_size*n_samples, hidden1_size, requires_grad=True)
#         h2_test = torch.rand(test_batch_size*n_samples, hidden2_size, requires_grad=True)
#         y_test = torch.rand(test_batch_size*n_samples, output_size, requires_grad=True)
#         target_test = train_target[idx:idx+test_batch_size].repeat(n_samples, 1).clone()
#         t_test = train_t[idx:idx+test_batch_size].repeat(n_samples).clone()

#     # Test error / accuracy
#     test_optimizer = optim.SGD([h1_test, h2_test, y_test], lr=mr)
#     h1_blowup, h2_blowup, y_blowup, energies = minimizeEnergy(model,5*free_steps,test_optimizer,x_test,h1_test,h2_test,y_test,print_energy=False)

#     print("Tests after randomizing internal state")
#     error = (y_blowup-target_test).pow(2).sum(dim=1).mean()
#     prediction = torch.argmax(y_blowup, dim=1)
#     accuracy = torch.mean((prediction==t_test).float())
#     print("Test error: ",error.item())
#     print("Test accuracy: ",accuracy.item())
#     print("Testing time: ",int((time.time()-tic)/60)," minutes")
# #####################################################################################################        

#     # print("")
#     # print("Max y val: ",torch.max(y_blowup))
#     # print("Min y val: ",torch.min(y_blowup))
#     # print("Max h1 val: ",torch.max(h1_blowup))
#     # print("Min h1 val: ",torch.min(h1_blowup))
#     # print("Max h2 val: ",torch.max(h2_blowup))
#     # print("Min h2 val: ",torch.min(h2_blowup))

#     # TSNE code
#     if args.make_tsne and (epoch+1)%2 == 0:
#         tic = time.time()
#         from sklearn.manifold import TSNE

#         def visualize_clusters(layer, title, colors=None, std=0.01, perplexity=30, cmap='tab10', s=2, alpha=0.1):
#             noise = std * torch.randn(layer.shape)
#             noisy_layer = layer + noise
#             X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(noisy_layer.numpy())
#             plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
#             plt.title(title)
#             plt.gca().set_aspect('equal')

#         # tSNE plot 1
#         plt.figure(figsize=(12, 10))  # You can adjust the dimensions as needed

#         for i,layer in enumerate([train_x,train_h1,train_h2,train_y]):
#             plt.subplot(2, 2, i+1)
#             visualize_clusters(layer,'Layer '+str(i), colors=train_t, perplexity=50)

#         cmap = plt.get_cmap('tab10') 
#         legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
#                         markerfacecolor=cmap(i), markersize=10) for i in range(10)]
#         plt.legend(handles=legend_handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.suptitle('t-SNE Visualization of Clusters\n Training initialization: '+args.train_init+'\nAccuracy: '+str(100*accuracy.item())+'%\nMSE: '+str(error.item()))
#         plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.9, hspace=0.2, wspace=0.05)
#         plt.savefig(args.output_dir+'/clusters1.png', bbox_inches='tight')

#         # tSNE plot 2
#         plt.figure(figsize=(12, 10))  # You can adjust the dimensions as needed

#         for i,layer in enumerate([x_test,h1_blowup,h2_blowup,y_blowup]):
#             plt.subplot(2, 2, i+1)
#             visualize_clusters(layer,'Layer '+str(i), colors=t_test, perplexity=50)
#         cmap = plt.get_cmap('tab10') 
#         legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
#                         markerfacecolor=cmap(i), markersize=10) for i in range(10)]
#         plt.legend(handles=legend_handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.suptitle('t-SNE Visualization of Clusters\n Training initialization: '+args.train_init+'\nAccuracy: '+str(100*accuracy.item())+'%\nMSE: '+str(error.item()))
#         plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.9, hspace=0.2, wspace=0.05)
#         plt.savefig(args.output_dir+'/clusters2.png', bbox_inches='tight')
#         print("tSNE plot time: ",int((time.time()-tic)/60)," minutes")