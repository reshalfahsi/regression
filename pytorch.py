import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class PolyRegression(nn.Module):
    """ Polynomial Regression module

    this will compute the polynomial regression with degree specified.
    for example, degree of 1 is the model for linear regression `y = a*x + b`,
    degree of 2 is the model for quadratic regression `y = a*(x^2) + b*x + c`, etc.

    Args:
        degree (int): the degree of polynomial regression

    """
    def __init__(self, degree):
        super(PolyRegression, self).__init__()
        self.degree = degree

        self.weight = nn.Parameter(torch.randn([degree+1, 1]))
    
    def regress_forward(self, x):
        x = torch.flatten(x)
        f = torch.stack([x.pow(a) for a in range(self.degree, -1, -1)], 1)
        y = f @ self.weight
        return y.squeeze()

    def forward(self, x):
        return self.regress_forward(x)
    
    def get_weight(self):
        return self.weight.detach().numpy().flatten()


def generate_data(num_iterate=1000, num_data=100):
    for _ in range(num_iterate):
        x = torch.rand(num_data) * 2 - 1            # interval of (0, 1]
        y = 0.5*x.pow(3) + 7*x.pow(2) + 5*x + 2
        yield x, y

def compute_r2(y, yreg):
    y = np.array(y)
    yreg = np.array(yreg)

    ss_tot = np.square(y - y.mean()).sum()
    ss_res = np.square(y - yreg).sum()
    return 1 - (ss_res/ss_tot)

if __name__ == "__main__":
    torch.manual_seed(123)
    
    model = PolyRegression(3)
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

    loss_data = []

    for x, y in generate_data(1000, num_data=1000):
        yreg = model(x)
        loss = criterion(yreg, y)

        loss_data.append(loss.detach().item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    print(model.get_weight())
    print("R2: ", compute_r2(y, yreg.detach()))
    # plt.plot(loss_data)
    # plt.show()
