import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats
from scipy.stats import kstest


# 2.0 Modeling the discrete random variable
n = 10000

xk = [2, 3, 5, 12, 21, 33, 44]
pk = [0.1, 0.02, 0.25, 0.15, 0.35, 0.03, 0.1]

steps = [0]
for i in range(len(pk)):
    steps.append(round(steps[i] + pk[i], 2))

R = np.random.uniform(0, 1, n)
new_pk = []
quantity = []
intervals_b = []
intervals_e = []

for i in range(len(steps)-1):
    intervals_b.append(steps[i])
    intervals_e.append(steps[i+1])
    x = R[(R > steps[i]) & (R < steps[i+1])]
    q = R[(R > steps[i]) & (R < steps[i+1])].size
    quantity.append(q)
    new_pk.append(q/n)


# 2.0 Plot the graph
plt.plot(R, color='black', markersize=0.05)
plt.xlabel('n')
plt.ylabel('X')
plt.show()

# 2.1 Finding mathematical expectation and the variance using Numpy functions
exp = np.mean(R)
vrc = np.var(R)
print(f'\nMathematical expectation: {round(float(exp), 8)}'
      f'\nThe variance: {round(float(vrc), 8)}')

# 2.2 Create the frequency table
frequency_table = pd.DataFrame({
    'Left': intervals_b,
    'Right': intervals_e,
    'Quantity': quantity,
    'Relative frequency': new_pk
    }, index=pd.RangeIndex(start=1, stop=8))

count = sum(quantity)

print(f'\n{frequency_table}')
print(f'              Sum: {count}')

# 2.3 Testing the hypothesis of the distribution law, creating a histogram
def test(data):
    d, p = kstest(data, 'uniform')
    print('p = ', p)
    return p > 0.05


print(f'The distribution is uniform: {test(R)}')

fig, ax = plt.subplots()
plt.hist(R, bins=steps, color='#ffff80', edgecolor='black')
plt.xticks(steps, rotation='vertical')
fig.set_figwidth(8)
fig.set_figheight(6)
plt.show()
