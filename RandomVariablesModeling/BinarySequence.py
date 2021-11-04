import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import kstest

n = 100001
mu = 0
sigma = 1

E = mu + sigma*np.random.standard_normal(size=n)
beta = np.array([])

for i in np.arange(n-1, step=1):
    if E[i+1] - E[i] > 0:
        beta = np.append(beta, [1])
    else:
        beta = np.append(beta, [0])

beta = beta.astype(int)

zero_one = np.bincount(beta)
zero, one = zero_one[0], zero_one[1]

freq_0 = zero / len(beta)
freq_1 = one / len(beta)

print(f'Amount of zero`s is {zero}\nAmount of one`s is {one}')
print(f'Frequency of zero`s is {freq_0}\nFrequency of one`s is {freq_1}')

# plt.plot(np.sort(beta), np.linspace(0, 1, len(beta), endpoint=False), color='black')
# plt.title('Graph of the empirical distribution function')
# plt.xlabel('Values')
# plt.ylabel('Distributive function')
# plt.show()
#
# fig, ax = plt.subplots()
# plt.bar([0, 1], zero_one/len(beta), tick_label=['zero', 'one'], color='#ffff80', width=0.6, edgecolor='black')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.show()

# plt.hist(beta, color='#ffff80')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.xticks([0, 1])
# plt.show()

def test(data):
    d, p = kstest(data, 'norm')
    print('p = ', p)
    return p > 0.05


print(f'The distribution is normal: {test(E)}')