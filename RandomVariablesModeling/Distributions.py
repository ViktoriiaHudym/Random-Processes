import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import cauchy
from scipy.stats import kstest
n = 1000000

# Gaussian distribution
def Gaussian():
    mu = float(input('Enter mu:'))
    sigma = float(input('Enter sigma:'))
    s_gauss = np.random.normal(mu, sigma, size=n)

    plt.scatter(range(n), s_gauss)
    plt.show()

    count, bins, ignored = plt.hist(s_gauss, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.title('Gaussian')
    plt.xlabel('x values')
    plt.ylabel('PDF')
    plt.show()

    return f'Mathematical expectation: {np.mean(s_gauss)}\n' \
           f'The variance: {np.var(s_gauss)}'


# Weibull distribution
def Weibull():
    k = float(input('Enter parameter k: '))
    l = float(input('Enter parameter lambda: '))
    s_weib = np.random.weibull(k, n)

    plt.plot(range(n), s_weib, color='black')
    plt.show()

    count, bins, ignored = plt.hist(s_weib, 30, density=True, color='#ffff80')
    plt.plot(bins, (k / l) * (bins / l) ** (k - 1) * np.exp(-(bins / l) ** k),
             linewidth=1, color='black')
    plt.title('Weibull')
    plt.show()

    def test(data):
        d, p = kstest(data, 'exponweib', [1, 1])
        print('p = ', p)
        return p > 0.05

    return f'Mathematical expectation: {np.mean(s_weib)}\nThe variance: {np.var(s_weib)}\n' \
           f'The distribution is normal: {test(s_weib)}'

def Rayleigh():
    sigma_r = float(input('Enter sigma: '))

    s_rayl = np.random.rayleigh(sigma_r, size=n)

    plt.plot(range(n), s_rayl, color='black')
    plt.show()

    count, bins, ignored = plt.hist(s_rayl, 30, density=True, color='#ffff80')
    plt.plot(bins, (bins / sigma_r ** 2) * np.exp((- bins ** 2) / (2 * sigma_r ** 2)),
             linewidth=1, color='black')
    plt.title('Rayleigh')
    plt.show()

    def test(data):
        d, p = kstest(data, 'rayleigh')
        print('p = ', p)
        return p > 0.05

    return f'Mathematical expectation: {np.mean(s_rayl)}\n' \
           f'The variance: {np.var(s_rayl)}\n' \
           f'The distribution is Rayleigh: {test(s_rayl)}'

def Lognormal():
    mu_l = float(input('Enter mu: '))
    sigma_l = float(input('Enter sigma: '))

    s_logn = np.random.lognormal(mu_l, sigma_l, size=n)

    plt.plot(range(n), s_logn, color='black')
    plt.show()

    count, bins, ignored = plt.hist(s_logn, 50, density=True, align='mid', color='#ffff80')
    pdf = (np.exp(-(np.log(bins) - mu_l) ** 2 / (2 * sigma_l ** 2))
           / (bins * sigma_l * np.sqrt(2 * np.pi)))
    plt.plot(bins, pdf, linewidth=1, color='black')
    plt.axis('tight')
    plt.title('Log-normal')
    plt.show()

    def test(data):
        d, p = kstest(data, 'lognorm', (1, 1))
        print('p = ', p)
        return p > 0.05

    return f'Mathematical expectation: {np.mean(s_logn)}\n' \
           f'The variance: {np.var(s_logn)}\n' \
           f'The distribution is Log-normal: {test(s_logn)}'

def Cauchy():
    gamma = float(input('Enter gamma: '))
    x = float(input('Enter x0: '))

    s_cauchy = cauchy.rvs(loc=x, scale=gamma, size=n)
    # s_cauchy = x + gamma *np.random.standard_cauchy(n)
    # s_cauchy = s_cauchy[(s_cauchy > -25) & (s_cauchy < 25)]  # truncate distribution so it plots well

    plt.plot(range(n), s_cauchy, color='black')
    plt.show()

    plt.hist(s_cauchy, bins=50, color='black')
    plt.show()

    def test(data):
        d, p = kstest(data, 'cauchy')
        print('p = ', p)
        return p > 0.05

    return f'Mathematical expectation: {np.mean(s_cauchy)}\n' \
           f'The variance: {np.var(s_cauchy)}\n' \
           f'The distribution is Cauchy: {test(s_cauchy)}'


if __name__ == '__main__':
    # print(Gaussian())
    # print(Weibull())
    # print(Rayleigh())
    # print(Lognormal())
    print(Cauchy())
    # choice = input('Choose law: \n  1 - Gaussian\n  2 - Weibull\n  3 - Relley'
    #                '\n  4 - log\n  5 - Koshi\n')
    # if choice == '1':
    #     print(Gaussian())
    # elif choice == '2':
    #     print(Weibull())
    # else:
    #     exit(1)

