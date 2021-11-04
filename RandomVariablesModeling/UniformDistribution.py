import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats


# 1.0 Generating a sequence of n pseudo-random numbers
n = 1000000
numbers = np.random.uniform(0, 1, n)

# # 1.0 Plot the graph
x = range(10000)
y = np.random.uniform(0, 1, size=10000)
plt.scatter(x, y, color='black', s=0.2)
plt.xlabel('Xn')
plt.ylabel('Xn+1')
plt.show()

# 1.1 Finding mathematical expectation and the variance
def calc_Expectation(a, m):
    prb = 1 / m
    result = 0
    for i in range(0, m):
        result += (a[i] * prb)
    return float(result)


def calc_Variance(a, m):
    result = 0
    for i in range(0, m):
        result += ((a[i]-expect)**2/m)
    return float(result)


expect = calc_Expectation(numbers.flat, n)
variance = calc_Variance(numbers.flat, n)
print(f'\nMathematical expectation E(X): {round(expect, 6)}\n'
      f'The variance Var(X): {round(variance, 6)}')

# 1.1 Finding mathematical expectation and the variance using Numpy functions
# exp = np.mean(numbers)
# vrc = np.var(numbers)
# print(f'\nMathematical expectation: {round(float(exp), 6)} \nThe variance: {round(float(vrc), 6)}')

# 1.2 Create the frequency table
L = round(1+3.322*math.log10(n))
h = round(1/L, 5)
print(f'\nAmount of intervals = {L} \nSampling step = {h}')

intervals_b = []
intervals_e = []
quantity = []
relative_frequency = []

for interval in np.arange(0, 1.0, h):
    intervals_b.append(interval)
    if (interval + h) > 1:
        intervals_e.append(1.0)
    else:
        intervals_e.append(interval + h)
    q = numbers[(interval < numbers) & (numbers < interval+h)].size
    quantity.append(q)
    relative_frequency.append(q/n)

count = sum(quantity)

frequency_table = pd.DataFrame({
    'Left': intervals_b,
    'Right': intervals_e,
    'Quantity': quantity,
    'Relative frequency': relative_frequency
    }, index=pd.RangeIndex(start=1, stop=L+1))

print(f'\n{frequency_table}')
print(f'              Sum: {count}')


# 1.3 Testing the hypothesis of the distribution law, creating a histogram

def calc_chi_observable(sample, interval_left, q):
    average = calc_Expectation(sample.flat, n)
    standart_deviation = math.sqrt(calc_Variance(sample.flat, n))
    a = average - math.sqrt(3) * standart_deviation
    b = average + math.sqrt(3) * standart_deviation
    dencity_function = 1 / (b - a)

    nq = sum(q)

    critical_freq2 = []
    if nq * dencity_function * (interval_left[1] - a) > 0:
        n1 = nq * dencity_function * (interval_left[1] - a)
        critical_freq2.append((q[0] - n1) ** 2 / n1)
    else:
        critical_freq2.append(0.0)

    for idx in range(1, len(q) - 1):
        ni = nq * (interval_left[idx] - interval_left[idx - 1]) \
             / (b - a)
        critical_freq2.append((q[idx] - ni) ** 2 / ni)

    ns = nq * dencity_function * (b - interval_left[-1])
    critical_freq2.append((q[-1] - ns) ** 2 / ns)

    return sum(critical_freq2)


def calc_chi_critical(alpha, s):
    result = scipy.stats.chi2.ppf(1 - alpha, s - 3)
    return result


def is_uniform_distribution():
    chi_observable = calc_chi_observable(numbers, intervals_b, quantity)
    chi_critical = calc_chi_critical(0.05, L)
    print(f'\nChi observable = {chi_observable}\nChi critical = {chi_critical}')
    if chi_observable < chi_critical:
        return f'Chi observable < Chi critical. Distribution IS uniform'
    else:
        return f'Chi observable > Chi critical. Distribution IS NOT uniform'


print(is_uniform_distribution())


fig, ax = plt.subplots()
plt.hist(numbers, bins=np.arange(0.0, 1.01, step=h), color='#ffff80', edgecolor='black')
plt.xticks(np.arange(0.0, 1.01, step=h), rotation='vertical')
fig.set_figwidth(10)
fig.set_figheight(6)
plt.show()

# 1.4
m = 1000
arr = []

for i in range(n):
    mumbers = np.random.uniform(0, 1, size=m)
    arr.append(np.max(mumbers))

plt.hist(arr, bins=30, color='#ffff80', edgecolor='black')
plt.show()


