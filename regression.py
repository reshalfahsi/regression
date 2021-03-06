import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from pytorch import compute_r2

available_regression = [
    'power_regression',
    'exponential_regression',
    'polynomial_regression',
    'linear_regression'
]


def power_regression(x, y):
    n = len(x)
    x_correction = 1 #1e-6
    y_correction = 1 #1e-6
    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    #if (x_mean != 0):
    #    power = np.log(x_mean)/np.log(10)
    #    x_correction *= pow(10, power)

    #if (y_mean != 0):
    #    power = np.log(y_mean)/np.log(10)
    #    y_correction *= pow(10, power)

    result = np.where(x == 0)
    for idx in result[0]: 
        #if (n > 1):
        #    if (idx < n-1): 
        #        x[idx] = (x[idx] + x[idx+1])/2.0
        #    else:
        #        x[idx] = (x[idx] + x[idx-1])/2.0
        #
        sign = np.sign(x[idx])
        if (sign>=0): sign = 1
        x[idx] += sign * x_correction

    result = np.where(y == 0)
    for idx in result[0]: 
        #if (n > 1):
        #    if (idx < n-1): 
        #        y[idx] = (y[idx] + y[idx+1])/2.0
        #    else:
        #        y[idx] = (y[idx] + y[idx-1])/2.0
        #
        sign = np.sign(y[idx])
        if (sign>=0): sign = 1
        y[idx] += sign * y_correction

    for i in range(n):
        term1 += np.log(abs(x[i])) * np.log(abs(y[i]))
        term2 += np.log(abs(x[i]))
        term3 += np.log(abs(y[i]))
        term4 += np.log(abs(x[i])) ** 2

    b = ((n*term1) - (term2* term3))/((n*term4)-(term2 ** 2))
    a = (term3 - (b*term2))/n
    a = np.exp(a)

    yreg = a*(x**b)
    r2 = compute_r2(y, yreg)

    print('Power Regression: y = a.x^(b)')
    print('a: ', a)
    print('b: ', b)
    print('R2: ', r2)

    x_max = np.max(x)
    x_min = np.min(x)
    x = np.linspace(x_min, x_max, 100)
    y = a*(x**b)

    return x, y, "y = {:.9f} * x^({:.6f})".format(a, b)

def exponential_regression(x, y):
    x_mean = np.mean(x)
    n = len(x)
    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0
    term5 = 0
    den = 0
    correction = 1 #1e-6
    #mean = np.mean(y)
    #if (mean != 0):
    #    power = np.log(mean)/np.log(10)
    #    correction *= pow(10, power)

    result = np.where(y == 0)
    for idx in result[0]: 
        #if (n > 1):
        #    if (idx < n-1): 
        #        y[idx] = (y[idx] + y[idx+1])/2.0
        #    else:
        #        y[idx] = (y[idx] + y[idx-1])/2.0
        #
        sign = np.sign(y[idx])
        if (sign>=0): sign = 1
        y[idx] += sign * correction

    for i in range(n):
        term1 += (x[i]*x[i]*y[i])
        term2 += (y[i]*np.log(abs(y[i])))
        term3 += (x[i]*y[i])
        term4 += (x[i]*y[i]*np.log(abs(y[i])))
        term5 += y[i]
    a = (term1*term2 - term3*term4)/(term5*term1 - term3*term3)
    a = np.exp(a)
    b = ((term5*term4) - term3*term2)/((term5*term1) - term3*term3)

    yreg = a * np.exp(b*x)
    r2 = compute_r2(y, yreg)

    print('Exponential Regression: y = a.e^(bx)')
    print('a: ', a)
    print('b: ', b)
    print('R2: ', r2)

    x_max = np.max(x)
    x_min = np.min(x)

    x = np.linspace(x_min, x_max, 100)
    y = a * np.exp(b*x)

    return x, y, r2, "y = {:.9f} * (e^({:.6f} * x))".format(a, b)

def polynomial_regression(x, y, degree=3):
    X = np.ones((len(x), degree+1))
    for n in range(1, degree+1):
        val = ((x**n) - np.mean(x**n))/(np.max(x**n) - np.min(x**n))
        X[:, n] = val

    m = len(x)
    theta = np.zeros(degree+1)
    yorig = np.array(y)

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

    x = np.linspace(np.min(x), np.max(x), 100)
    y = theta[0]
    yreg = theta[0]
    txt = "y = {:.4f}".format(theta[0])

    print('Polynomial Regression: a0 + a1.X + ... + an.X^n') 
    print("a0: %s" % theta[0])
    for n in np.arange(1, len(theta)):
        print('a%s: %s' % (n, theta[n]))
        txt += " + {:.4f}*x".format(theta[n]) if theta[n] >= 0 else \
            " - {:.4f}*x".format(-theta[n])
        if n > 1:
            txt += "^%s" % str(n)
        y += theta[n] * (x ** (n))
        yreg += theta[n] * (x**n)
    r2 = compute_r2(yorig, yreg)
    print('R2: ', r2)
    return x, y, r2, txt

def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = sum((x_i - x_mean)*(y_i - y_mean) for (x_i, y_i) in zip(x, y))
    den = sum((x_i - x_mean)**2 for x_i in x)

    m = num/den
    c = y_mean - (m * x_mean)

    yreg = m*x + c
    r2 = compute_r2(y, yreg)

    print('Linear Regression: y = mx + c')
    print('m: ', m)
    print('c: ', c)
    print('R2: ', r2)

    x = np.linspace(np.min(x), np.max(x), 100)
    y = c + m * x
    return x, y, r2, "y = ({:.4f})*x + ({:.4f})".format(m, c)

def read_csv(filename):
    with open(filename) as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=',')
        data = [row for row in reader]
        data = {k: v for k, v in zip(data[0], np.array(data[1:], dtype=np.float).T)}
    return data

def regression(args):
    types = args.type + '_regression' if 'regression' not in args.type else args.type
    data = read_csv(str(args.dataset))
    x, y = data['x'], data['y']

    if types not in available_regression:
        raise RuntimeError("Invalid Regression Type of '%s', available: %s" % (types, 
                           ", ".join(available_regression)))
    kwargs = {}
    if types == 'polynomial_regression':
        kwargs['degree'] = args.degree

    x_reg, y_reg, r2, text = eval(f"{types}(x, y, **kwargs)")
    
    label = ' '.join(word.capitalize() for word in types.replace('_', ' ').split())
    plt.plot(x_reg, y_reg, color='#a7de77', label=label)
    plt.scatter(x, y, color='#9b354c', label='Data Point')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(label)
    plt.legend()
    x_lim = plt.gca().get_xlim()
    y_lim = plt.gca().get_ylim()
    x_text = x_lim[0] + ((x_lim[1] - x_lim[0])/64)
    y_text = y_lim[1] - ((y_lim[1] - y_lim[0])/5)
    plt.text(x_text, y_text, text)
    r2_txt = "R2: {:.4}".format(r2)
    plt.text(x_text, y_text-((y_lim[1] - y_lim[0])/15), r2_txt)
    plt.show()

if __name__=='__main__':
    regression_types = [reg.replace('_regression', '') for reg in available_regression]
    parser = argparse.ArgumentParser()
    parser.add_argument('type', nargs='?', default='linear', 
        help='regression type to be executed', choices=regression_types)
    parser.add_argument('-d', '--dataset', default='dataset/default.csv', 
        help='dataset path')
    parser.add_argument('--degree', default=3, type=int,
        help='degree of polynomial regression')
    args = parser.parse_args()
    regression(args)
