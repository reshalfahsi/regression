import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import math
from collections import OrderedDict


available_regression = [
    'power_regression',
    'exponential_regression',
    'polynomial_regression',
    'linear_regression'
]


def power_regression(x, y):
    n = len(x)
    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0
    
    for i in range(n):
        term1 += np.log(x[i]) * np.log(y[i])
        term2 += np.log(x[i])
        term3 += np.log(y[i])
        term4 += np.log(x[i]) ** 2

    b = ((n*term1) - (term2* term3))/((n*term4)-(term2 ** 2))
    a = (term3 - (b*term2))/n
    a = np.exp(a)

    print('Power Regression: y = a.x^(b)')
    print('a: ', a)
    print('b: ', b)

    x_max = np.max(x)
    x_min = np.min(x)
    x = np.linspace(x_min, x_max, 100)
    y = a*(x**b)

    return x, y

def exponential_regression(x, y):
    x_mean = np.mean(x)
    n = len(x)
    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0
    term5 = 0
    den = 0
    for i in range(n):
        term1 += (x[i]*x[i]*y[i])
        term2 += (y[i]*np.log(y[i]))
        term3 += (x[i]*y[i])
        term4 += (x[i]*y[i]*np.log(y[i]))
        term5 += y[i]
    a = (term1*term2 - term3*term4)/(term5*term1 - term3*term3)
    a = np.exp(a)
    b = (term5*term4 - term3*term2)/(term5*term1 - term3*term3)

    print('Exponential Regression: y = a.e^(bx)')
    print('a: ', a)
    print('b: ', b)

    x_max = np.max(x)
    x_min = np.min(x)

    x = np.linspace(x_min, x_max, 100)
    y = a * np.exp(b*x)

    return x, y

def polynomial_regression(x, y, degree=3):
    x_mean = np.mean(x)
    x_max =  np.max(x)
    x_min = np.min(x)
    d = {}
    d['x' + str(0)] = np.ones([1,len(x)])[0]
    for n in np.arange(1, degree+1):
        d['x' + str(n)] = ((x**n) - np.mean(x**n))/(np.max(x**n) - np.min(x**n))
    d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    X = np.column_stack(d.values())

    m = len(x)
    theta = np.zeros(degree+1)

    h = theta[0]
    for n in np.arange(1, len(theta)):
        h += theta[n] * (x ** n)
    error = h - y

    for i in range(100000):
        #gradient descent
        theta += (-0.001) * (1/m) * np.dot(error, X)

        h = theta[0]
        for n in np.arange(1, len(theta)):
            h += theta[n] * (x ** n)
        error = h - y
        cost = (1/(2*m)) * np.sum(error**2)
        if cost < 0.001:
            theta += (-0.001) * (1/m) * np.dot(error, X)
            break

    x = np.linspace(x_min, x_max, 100)
    y = theta[0]

    print('Polynomial Regression: a0 + a1.X + ... + an.X^n') 
    for n in np.arange(1, len(theta)):
        print('a'+ str(n),': ',theta[n])
        y += theta[n] * (x ** (n))

    return x, y

def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)
    num = 0
    den = 0
    for i in range(n):
        num += (x[i] - x_mean) * (y[i] - y_mean)
        den += (x[i] - x_mean) ** 2

    b1 = num/den
    b0 = y_mean - (b1 * x_mean)

    print('Linear Regression: y = mx +b')
    print('m: ', b1)
    print('b: ', b0)

    x_max = np.max(x)
    x_min = np.min(x)

    x = np.linspace(x_min, x_max, 100)
    y = b0 + b1 * x
    return x, y

def read_csv(filename):
    with open('dataset.csv') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=',')
        data = [row for row in reader]
        data = {k: v for k, v in zip(data[0], np.array(data[1:], dtype=np.float).T)}
    return data

def regression(types):
    data = read_csv('dataset.csv')
    x, y = data['x'], data['y']

    if types not in available_regression:
        raise RuntimeError("Invalid Regression Type of '%s', available: %s" % (types, 
                           ", ".join(available_regression)))
    kwargs = {}
    if types == 'polynomial_regression':
        kwargs['degree'] = 3
    
    x_reg, y_reg = eval(f"{types}(x, y, **kwargs)")
    
    label = ' '.join(word.capitalize() for word in types.replace('_', ' ').split())
    plt.plot(x_reg, y_reg, color='#a7de77', label=label)
    plt.scatter(x, y, color='#9b354c', label='Data Point')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(label)
    plt.legend()
    plt.show()

if __name__=='__main__':
    types = 'linear_regression'
    if(len(sys.argv)>1):
        types = str(sys.argv[1]) 
    regression(types)
