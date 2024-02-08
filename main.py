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
parser.add_argument("--init", type=str, default="zeros", help="Initialization method for weights")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
parser.add_argument("--mr", type=float, default=0.5, help="Energy minimization rate")
parser.add_argument("--lam", type=float, default=2.0, help="Lambda value for weight updates")
parser.add_argument("--make-tsne", action="store_true", help="Make t-SNE plot")
parser.add_argument("--print-frequency", type=int, default=25, help="Frequency of printing the error")
parser.add_argument("--n-epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--output-file", type=str, default="results", help="Output csv file name")
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
output_file = args.output_file
test_batch_size = batch_dim


if args.dataset == "mnist":
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,)),transforms.Lambda(lambda x: x.view(-1))])
    trainset = torchvision.datasets.MNIST(root='~/datasets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_dim, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='~/datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


# Define the fixed random input x
elif args.dataset == "random":
    x = torch.randn(batch_dim,input_size)
    t = torch.tensor([i%output_size for i in range(batch_dim)],dtype=torch.long)
    target = torch.nn.functional.one_hot(t, num_classes=output_size)

# elif args.dataset == "mnist":
#     for batch in trainloader: # Get one MNIST batch
#         x,t = batch
#         break
else:
    raise ValueError("Invalid dataset")




# Define the optimizer

# For weight updates
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

# Optimization loop
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

errors = []

h1 = torch.zeros(batch_dim, hidden1_size, requires_grad=True)
h2 = torch.zeros(batch_dim, hidden2_size, requires_grad=True)
y = torch.zeros(batch_dim, output_size, requires_grad=True)
optimizer = optim.SGD([h1, h2, y], lr=mr)

###############
# Training loop
###############
for epoch in range(n_epochs):
    print("Epoch: ",epoch)
    for itr,batch in enumerate(trainloader): # MNIST
        if args.dataset == "mnist":
            x,t = batch
        else:
            pass # Use the fixed random batch

        target = torch.nn.functional.one_hot(t, num_classes=10)

        # Initialize internal state variables
        if args.init == "zeros":
            h1.data.zero_()
            h2.data.zero_()
            y.data.zero_()
        elif args.init == "random":
            h1.data.uniform_(0,1)
            h2.data.uniform_(0,1)
            y.data.uniform_(0,1)
        else:
            raise ValueError("Invalid initialization method")

        # optimizer = optim.SGD([h1, h2, y], lr=mr)
        
        h1_free, h2_free, y_free, phase1_energies = minimizeEnergy(model,free_steps,optimizer,x,h1,h2,y,print_energy=False)

        if (itr+1)%(n_iters//print_frequency) == 0:
            print("\nIteration: ",itr+1)
            # print("Output: ",y_free)
            print("Error: ",(y_free-target).pow(2).sum())
            prediction = torch.argmax(y_free, dim=1)
            accuracy = torch.mean((prediction==t).float())
            if accuracy < 0.5:
                assert(0)   # Kill if not learning
            print("Accuracy: ",accuracy)

        errors.append((y_free-target).pow(2).sum())

        h1_nudge, h2_nudge, y_nudge, phase2_energy = minimizeEnergy(model,nudge_steps,optimizer,x,h1,h2,y,target=target,print_energy=False)
        energies = phase1_energies + phase2_energy
        
        # Calculate the weight updates
        w1_update = (x.t() @ h1_nudge - x.t() @ h1_free)/(beta*batch_dim)
        w2_update = (h1_nudge.t() @ h2_nudge - h1_free.t() @ h2_free)/(beta*batch_dim)
        w3_update = (h2_nudge.t() @ y_nudge - h2_free.t() @ y_free)/(beta*batch_dim)
    
        b1_update = (h1_nudge - h1_free).sum(0)/(beta*batch_dim)
        b2_update = (h2_nudge - h2_free).sum(0)/(beta*batch_dim)
        b3_update = (y_nudge - y_free).sum(0)/(beta*batch_dim)

        # # Print L2 norm of weight updates
        # if (itr+1)%(n_iters//print_frequency) == 0:
        #     print("")
        #     print("L2 norm of weight updates (w1):", torch.norm(w1_update).item())
        #     print("L2 norm of weight updates (w2):", torch.norm(w2_update).item())
        #     print("L2 norm of weight updates (w3):", torch.norm(w3_update).item())
        #     print("L2 norm of weight updates (b1):", torch.norm(b1_update).item())
        #     print("L2 norm of weight updates (b2):", torch.norm(b2_update).item())
        #     print("L2 norm of weight updates (b3):", torch.norm(b3_update).item())

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
        if (itr+1)%(n_iters//print_frequency) == 0:
            # Plot the error
            plt.figure()
            plt.plot(errors)
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title('Error vs. Iteration\nbeta: '+str(beta)+'\nlearning_rate: '+str(learning_rate)+'\nError: '+str(errors[-1]))
            plt.savefig('error_vs_iteration_beta_'+str(beta)+'_lr_'+str(learning_rate)+'.png')
            plt.close()

    # Testing
    ######################################################################################################
    test_errors, test_accuracies = [],[]
    for itr,batch in enumerate(testloader):
        x_test,t_test = batch
        target_test = torch.nn.functional.one_hot(t_test, num_classes=10)
        if args.init == "zeros":
            h1.data.zero_()
            h2.data.zero_()
            y.data.zero_()
        elif args.init == "random":
            h1.data.uniform_(0,1)
            h2.data.uniform_(0,1)
            y.data.uniform_(0,1)
        h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x_test,h1,h2,y,print_energy=False)
        error = (y_free-target_test).pow(2).sum()
        prediction = torch.argmax(y_free, dim=1)
        accuracy = torch.mean((prediction==t_test).float())
        test_errors.append(error.item())
        test_accuracies.append(accuracy.item())
    print("Test accuracy: ",np.mean(test_accuracies))
    print("Test error: ",np.mean(test_errors))

    # Write test error to CSV file
    with open(output_file+'.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['Test Error'])
        writer.writerow([epoch,np.mean(test_errors)])

    # if args.make_tsne:
    #     if args.dataset == "mnist":
    #         for itr,batch in enumerate(testloader): # MNIST
    #             if np.random.random()<0.1:
    #                 x_test,t_test = batch
    #                 target_test = torch.nn.functional.one_hot(t_test, num_classes=10)
    #                 break
    #     elif args.dataset == "random":
    #         x_test, t_test, target_test = x, t, target
    #     else:
    #         raise ValueError("Invalid dataset")
        
    #     # Generate random batch of points
    #     n_samples = 1000

    #     x_test = x_test.repeat(n_samples,1).clone()
    #     target_test = target_test.repeat(n_samples,1).clone()
    #     t_test = t_test.repeat(n_samples).clone()

    #     h1_test = torch.rand(test_batch_size*n_samples, hidden1_size, requires_grad=True)
    #     h2_test = torch.rand(test_batch_size*n_samples, hidden2_size, requires_grad=True)
    #     y_test = torch.rand(test_batch_size*n_samples, output_size, requires_grad=True)

    #     optimizer = optim.SGD([h1_test, h2_test, y_test], lr=mr)

    #     h1_free, h2_free, y_free, energies = minimizeEnergy(model,free_steps,optimizer,x_test,h1_test,h2_test,y_test,print_energy=False)

    #     print("Max y val: ",torch.max(y_free))
    #     print("Min y val: ",torch.min(y_free))
    #     print("Max h1 val: ",torch.max(h1_free))
    #     print("Min h1 val: ",torch.min(h1_free))
    #     print("Max h2 val: ",torch.max(h2_free))
    #     print("Min h2 val: ",torch.min(h2_free))
 
    #     from sklearn.manifold import TSNE

    #     # colors = t_test # [t_test[i % batch_dim] for i in range(4 * n_samples)]  # Example color vector
    #     colors = [i%test_batch_size for i in range(test_batch_size*n_samples)]
    #     cmap = plt.get_cmap('tab10') #cm.get_cmap('viridis')
    #     # s, alpha = 2, 1.0
    #     def visualize_clusters(layer, colors, std, title, perplexity, cmap=cmap, s=2, alpha=0.1):
    #         noise = std * torch.randn(layer.shape)
    #         noisy_layer = layer + noise
    #         X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(noisy_layer.numpy())
    #         plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=s, alpha=alpha, cmap=cmap)
    #         plt.title(title)
    #         plt.gca().set_aspect('equal')

    #     plt.figure(figsize=(12, 10))  # You can adjust the dimensions as needed

    #     plt.subplot(2, 2, 1)
    #     visualize_clusters(h1_free, colors, 0.01, 'hidden 1', 50)

    #     plt.subplot(2, 2, 2)
    #     visualize_clusters(h2_free, colors, 0.01, 'hidden 2', 50)

    #     plt.subplot(2, 2, 3)
    #     visualize_clusters(y_free, colors, 0.01, 'output', 50)

    #     plt.subplot(2, 2, 4)
    #     visualize_clusters(x_test, colors, 0.01, 'input', 50)

    #     legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
    #                     markerfacecolor=cmap(i), markersize=10) for i in range(10)]
    #     plt.legend(handles=legend_handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.suptitle('t-SNE Visualization of Clusters\n Training init: zeros')

    #     plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.9, hspace=0.2, wspace=0.05)
    #     plt.savefig('clusters.png', bbox_inches='tight')
