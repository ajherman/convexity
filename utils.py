import torch
import torch.nn as nn

class HopfieldEnergy(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, beta=1.0):
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

        self.beta = beta

        # self.F = nn.MSELoss()


    # def forward(self, x, h1, h2, y, target=None): # Testing

    #     # Hopfield energy (layered)
    #     # energy1 = -(x @ self.w1 @ h1.t() + self.b1 @ h1.t()) 
    #     # energy2 = -(h1 @ self.w2 @ h2.t() + self.b2 @ h2.t())
    #     # energy3 = -(h2 @ self.w3 @ y.t() + self.b3 @ y.t())

    #     energy1 = (-self.w1(x, h1) - self.b1(h1)).squeeze()
    #     energy2 = (-self.w2(h1, h2) - self.b2(h2)).squeeze()
    #     energy3 = (-self.w3(h2, y) - self.b3(y)).squeeze()

    #     print(energy1.shape)
    #     print(energy2.shape)
    #     print(energy3.shape)
    #     # L2 penalty
    #     l2 = h1.pow(2).sum(dim=1) + h2.pow(2).sum(dim=1) + y.pow(2).sum()
    #     # l2 = h1.pow(2).sum() + h2.pow(2).sum() + y.pow(2).sum() # Old version
    #     print(l2.shape)
    #     # Total energy
    #     E = energy1 + energy2 + energy3 + 10*l2
    #     print(E.shape)
    #     # Nudge energy
    #     if target is not None:
    #         F = 10*self.beta*0.5*(y-target).pow(2).sum() #self.F(y, target)
    #         E += F
    #     print(E.shape)
    #     assert(0)
    #     return E.sum() #E.mean() # 

    def forward(self, x, h1, h2, y, target=None):

        # Hopfield energy (layered)
        energy1 = (-self.w1(x, h1) - self.b1(h1)).squeeze()
        energy2 = (-self.w2(h1, h2) - self.b2(h2)).squeeze()
        energy3 = (-self.w3(h2, y) - self.b3(y)).squeeze()

        # L2 penalty
        l2 = h1.pow(2).sum(dim=1) + h2.pow(2).sum(dim=1) + y.pow(2).sum(dim=1)
        # l2 = h1.pow(2).sum() + h2.pow(2).sum() + y.pow(2).sum() # Old version

        # Total energy
        E = energy1 + energy2 + energy3 + 0.5*l2

        # Nudge energy
        if target is not None:
            F = self.beta*0.5*(y-target).pow(2).sum(dim=1) #self.F(y, target)
            E += F

        return E.sum() #E.mean() # I think sum makes more sense for minimizing energy. We aren't really trying to average over the batch dimension.

class HopfieldUpdate(nn.Module):
    def __init__(self,beta=1.0):
        super(HopfieldUpdate, self).__init__()
        self.beta = beta
    def forward(self, x_initial, y_initial, x_final, y_final):
        out = torch.outer(x_final,y_final) - torch.outer(x_initial,y_initial)
        return out/self.beta
    


    
