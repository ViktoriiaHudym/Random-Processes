import numpy as np
import math


def f1(x):
    return x**7+x**5+x**3

def f2(x):
    return 2*math.sin(3*x)

def f3(x):
    return 1/((x+1)*math.sqrt(x))

def monte_carlo(f, a, b, n):
    mult_value = (b-a)/n
    sum = 0
    for i in range(1, n):
        x = np.random.uniform(a, b)
        sum += f(x)
    return mult_value*sum


n = 1000000

print(f'Integral #1 is equal to: {monte_carlo(f1, 0, 1, n)}\n'
      f'Integral #2 is equal to: {monte_carlo(f2, 0, math.pi, n)}\n'
      f'Integral #3 is equal to: {monte_carlo(f3, 0, 999, n)}')
