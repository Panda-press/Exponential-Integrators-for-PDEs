import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.special import factorial
import matplotlib.pyplot as plt

def step(t, x, tau, A, R, order = 1):
    result = 0
    result += np.exp(tau * A(t, x)) * x
    if order == 1:
        result += tau * phi(tau * A(t, x), 1) * R(t, x)
    elif order == 2:
        result += tau * ((phi(tau * A(t, x), 1) - 1/0.5 * phi(tau * A(t, x), 2)) * R(t, x) \
                         + 1/0.5 * phi(tau * A(t, x), 2) * R(t + 0.5 * tau, np.exp(tau * 0.5 * A(t, x)) * x + 0.5 * tau * phi(0.5 * tau * A(t, x), 1) * R(t, x)))
    return result

def phi(z, k):
    func = lambda theta: (np.exp((1-theta)* z) * (theta)**(k-1)/factorial(k-1))
    return integrate.quad(func, 0, 1)[0]
if True:
    def A(t, x):
        return 2*x

    def dxdt(t, x):
        return x*x

    x_0 = 0.1
    c = 1/x_0
    true = lambda t: 1/(c-t)
    num = 4
    addition = 7
    end_time = 5
else:
    def A(t, x):
        return 0
    def dxdt(t, x):
        return np.cos(t)
    x_0 = 1.1
    c = x_0
    true = lambda t: np.sin(t) + c
    num = 4
    addition = 8
    end_time = 0.5

def R(t, x):
    return dxdt(t, x) - A(t, x) * x

taus = []
errors1 = []
errors2 = []
for N in [2**i for i in range(num, num+addition)]:
    print(N)
    time = 0
    x = x_0
    x_fe = x_0
    tau = end_time / N
    taus.append(tau)
    for n_i in range(0, N):
        x = step(time, x, tau, A, R)
        x_fe += tau * (R(time, x_fe) + A(time, x_fe) * x_fe)
        time += tau
        print(f"Time:   {time}")
        print(f"Approx1:{x_fe}")
        print(f"Approx2:{x}")
        print(f"True:   {true(time)}")
    errors1.append(abs(x_fe - true(time)))
    errors2.append(abs(x - true(time)))

print(f"FE error \n {errors1}")
print(f"EI error \n {errors2}")

eocs1 = []
eocs2 = []
for i in range(0, len(taus)-1):
    eocs1.append(np.log(errors1[i]/errors1[i+1])/np.log(taus[i]/taus[i+1]))
    eocs2.append(np.log(errors2[i]/errors2[i+1])/np.log(taus[i]/taus[i+1]))
print(f"FE EOC: \n {eocs1}")
print(f"EI EOC: \n {eocs2}")

