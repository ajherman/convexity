import torch
import torch.nn as nn
import torch.optim as optim
from utils import HopfieldEnergy, HopfieldUpdate
import torch
import argparse
import matplotlib.pyplot as plt
# import torchvision
# import torchvision.transforms as transforms

# # Load the MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

parser = argparse.ArgumentParser()
parser.add_argument("--input-size", type=int, default=50, help="Size of the input")
parser.add_argument("--hidden1-size", type=int, default=100, help="Size of the first hidden layer")
parser.add_argument("--hidden2-size", type=int, default=100, help="Size of the second hidden layer")
parser.add_argument("--output-size", type=int, default=7, help="Size of the output")
parser.add_argument("--free-steps", type=int, default=40, help="Number of free optimization steps")
parser.add_argument("--nudge-steps", type=int, default=5, help="Number of nudge optimization steps")
parser.add_argument("--learning-rate", type=float, default=20.0, help="Learning rate for optimization")
parser.add_argument("--beta", type=float, default=4.0, help="Beta value for weight updates")
parser.add_argument("--batch-dim", type=int, default=7, help="Batch dimension")
parser.add_argument("--n-iters", type=int, default=1000, help="Number of iterations for optimization")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--init", type=str, default="random", help="Initialization method for weights")
args = parser.parse_args()

input_size = args.input_size
hidden1_size = args.hidden1_size
hidden2_size = args.hidden2_size
output_size = args.output_size
free_steps = args.free_steps
nudge_steps = args.nudge_steps
learning_rate = args.learning_rate
beta = args.beta
batch_dim = args.batch_dim
n_iters = args.n_iters

# Define the fixed random input x
x = torch.randn(batch_dim,input_size)
# x = (torch.rand(batch_dim,input_size)<0.02).float()*torch.randn(batch_dim,input_size)
target = torch.zeros(batch_dim,output_size)
for idx in range(batch_dim):
    target[idx,idx]=1
# print(target)
# assert(0)

# Define the optimizer

# For weight updates
model = HopfieldEnergy(input_size, hidden1_size, hidden2_size, output_size, beta=beta)


# Optimization loop
print("beta: ",beta)
print("learning_rate: ",learning_rate)
error = []
for itr in range(n_iters):
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

    optimizer = optim.SGD([h1, h2, y], lr=0.1)

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

    if (itr+1)%(n_iters//20) == 0:
        # print("Output: ",y_free)
        print("Error: ",(y_free-target).pow(2).sum())

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

    # if itr%100==0:
    #     # Plot the energies
    #     plt.plot(energies,label=str(itr))
    #     plt.xlabel('Step')
    #     plt.ylabel('Energy')
    #     plt.title('Energy vs. Step')
    #     plt.legend()
    #     plt.savefig('energy_vs_step.png')

    #     # Plot the error
    #     plt.figure()
    #     plt.plot(error)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Error')
    #     plt.title('Error vs. Iteration\nbeta: '+str(beta)+'\nlearning_rate: '+str(learning_rate)+'\nError: '+str(error[-1]))
    #     plt.savefig('error_vs_iteration_beta_'+str(beta)+'_lr_'+str(learning_rate)+'.png')
    #     plt.close()
            
        