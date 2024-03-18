import torch
import torch.nn as nn
import numpy as np

class HopfieldEnergy(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, beta=1.0, lam=1.0):
        super(HopfieldEnergy, self).__init__()
        # self.w1 = nn.Parameter(torch.randn(input_size,hidden1_size))
        # self.b1 = nn.Parameter(torch.randn(hidden1_size))
        # self.w2 = nn.Parameter(torch.randn(hidden1_size,hidden2_size))
        # self.b2 = nn.Parameter(torch.randn(hidden2_size))
        # self.w3 = nn.Parameter(torch.randn(hidden2_size,output_size))
        # self.b3 = nn.Parameter(torch.randn(output_size))

        # Equivalent of above using nn.bilinear
        self.w1 = nn.Bilinear(input_size, hidden1_size, 1, bias=False)
        self.b1 = nn.Linear(hidden1_size, 1, bias=False)
        self.w2 = nn.Bilinear(hidden1_size, hidden2_size, 1, bias=False)
        self.b2 = nn.Linear(hidden2_size, 1, bias=False)
        self.w3 = nn.Bilinear(hidden2_size, output_size, 1, bias=False)
        self.b3 = nn.Linear(output_size, 1, bias=False)

        # self.beta = beta
        self.lam = lam

    def forward(self, x, h1, h2, y, target=None, beta=None):
        # Hopfield energy (layered)
        energy1 = (-self.w1(x, h1) - self.b1(h1)).squeeze()
        energy2 = (-self.w2(h1, h2) - self.b2(h2)).squeeze()
        energy3 = (-self.w3(h2, y) - self.b3(y)).squeeze()

        # L2 penalty
        l2 = h1.pow(2).sum(dim=1) + h2.pow(2).sum(dim=1) + y.pow(2).sum(dim=1)

        # Total energy
        E = energy1 + energy2 + energy3 + self.lam*0.5*l2

        # Nudge energy
        # print(E.shape)
        if target is not None:
            F = (beta*0.5*(y-target).pow(2)).sum(dim=1) #self.F(y, target)
            # print(beta.shape)
            # print(y.shape)
            # print(target.shape)
            # print(F.shape)
            # assert(0)
            E += F

        return E.sum() 
    
class HopfieldUpdate(nn.Module):
    def __init__(self,beta=1.0):
        super(HopfieldUpdate, self).__init__()
        self.beta = beta
    def forward(self, x_initial, y_initial, x_final, y_final):
        out = torch.outer(x_final,y_final) - torch.outer(x_initial,y_initial)
        return out/self.beta
    


    
