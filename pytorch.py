import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class PolyRegression(nn.Module):
    """ Polynomial Regression module

    this will compute the polynomial regression with degree specified.
    for example, degree of 1 is the model for linear regression `y = a*x + b`,
    degree of 2 is the model for quadratic regression `y = a*(x^2) + b*x + c`, etc.

    keep in mind to always normalise, with interval of (0, 1], the data (`x` variable)
    you input to the model, if you don't want to cause a gradient explosion because 
    the output value is too big to handle.

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


def compute_r2(y, yreg):
    if isinstance(y, torch.Tensor):
        y = np.array(y.detach())
    if isinstance(yreg, torch.Tensor):
        yreg = np.array(yreg.detach())

    ss_tot = np.square(y - y.mean()).sum()
    ss_res = np.square(y - yreg).sum()
    return 1 - (ss_res/ss_tot)

def normalise(x):
    return (x - x.mean()) / x.std()

def fit(model, x, y, iteration=1000):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x) 
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

    loss_data = []
    for _ in range(iteration):
        yreg = model(x)
        loss = criterion(yreg, y)

        loss_data.append(loss.detach().item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    return model.get_weight(), loss_data


if __name__ == "__main__":
    torch.manual_seed(123)      # uncomment this if you want a non-deterministic result

    x = normalise(torch.randn(10))
    y = 0.5*x.pow(3) + 7*x.pow(2) + 5*x + 2
    degree = 3

    model = PolyRegression(degree)
    weight, loss_data = fit(model, x, y)

    yreg = model(x)
    print(weight)
    print("R2: ", compute_r2(y, yreg))

    ## plot regression
    x_reg = torch.tensor(np.linspace(x.min(), x.max(), 200, endpoint=False), dtype=torch.float32)
    y_reg = model(x_reg).detach().numpy()
    x_reg = x_reg.numpy()
    plt.plot(x_reg, y_reg, color='#a7de77', label=f'Polynomial Regression, degree {degree}')
    plt.scatter(x, y, color='#9b354c', label='Data Point')
    plt.legend()
    # plt.plot(loss_data)
    plt.show()
