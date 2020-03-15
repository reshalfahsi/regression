import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from collections import OrderedDict
import math

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

def polynomial_regression(x, y, degree):
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

def regression(types):
    dataset = pd.read_csv('dataset.csv')
    x = dataset['x'].values
    y = dataset['y'].values

    if(types=='linear_regression'):
        x_lr, y_lr = linear_regression(x, y)
        plt.plot(x_lr, y_lr, color='#a7de77', label='Linear Regression')
        plt.scatter(x, y, color='#9b354c', label='Data Point')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
    elif(types=='polynomial_regression'):
        x_poly, y_poly = polynomial_regression(x, y, 3)
        plt.plot(x_poly, y_poly, color='#a7de77', label='Polynomial Regression')
        plt.scatter(x, y, color='#9b354c', label='Data Point')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polynomial Regression')
        plt.legend()
        plt.show()
    elif(types=='power_regression'):
        x_pow, y_pow = power_regression(x, y)
        plt.plot(x_pow, y_pow, color='#a7de77', label='Power Regression')
        plt.scatter(x, y, color='#9b354c', label='Data Point')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Power Regression')
        plt.legend()
        plt.show()
    elif(types=='exponential_regression'):
        x_exp, y_exp = exponential_regression(x, y)
        plt.plot(x_exp, y_exp, color='#a7de77', label='Exponential Regression')
        plt.scatter(x, y, color='#9b354c', label='Data Point')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Exponential Regression')
        plt.legend()
        plt.show()
    else:
        print('Invalid Type of Regression!!')

if __name__=='__main__':
    types = 'linear_regression'
    if(len(sys.argv)>1):
        types = str(sys.argv[1]) 
    regression(types)
