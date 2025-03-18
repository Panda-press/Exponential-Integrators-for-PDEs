import numpy as np
import pandas as pd
import time as time
from numpy.linalg import norm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import diags, csc_matrix
from Arnoldi import ArnoldiExp
from Stable_Lanzcos import LanzcosExp
# from NBLA import NBLAExp
# from kiops import KiopsExp

# Create matrix A with given tau
def GetA(n, tau=1):
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    A = csc_matrix(A)
    return -A * tau

# Create initial vector v
def GetV(n):
    v = np.sin(np.linspace(0, np.pi, n)) + 1
    return v/np.linalg.norm(v)

# Define methods to test
methods = {
    "Scipy": lambda A, v, m: expm_multiply(A, v),
    "Arnoldi": ArnoldiExp,
    "Lanzcos": LanzcosExp
}

m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
m = [i for i in range(1, 64)]
n = [2**i for i in range(0, 16)]
tau_vals = [10**-i for i in range(0, 3)]

# Create DataFrame for results
data = pd.DataFrame(columns=["N", "M", "Method", "Tau", "Error", "Computation Time"])

# Loop over all tau values
for tau in tau_vals:
    true_values = {}
    scipy_computing_time = {}

    for N in n:
        print(f"Computing true values for N={N}, Tau={tau}")
        A = GetA(N, tau)
        V = GetV(N)
        start = time.time()
        result = methods["Scipy"](A, V, 1)
        end = time.time()
        true_values[N] = result
        scipy_computing_time[N] = end - start

    # Iterate over all combinations of N, M, and methods
    for N in n:
        for M in m:
            for method_name, method in methods.items():
                if N < M:
                    # Skip if M is too large for N
                    continue

                print(f"Method: {method_name}, N={N}, M={M}, Tau={tau}")
                A = GetA(N, tau)
                V = GetV(N)

                if method_name == "Scipy":
                    error = 0.0
                    comp_time = scipy_computing_time[N]
                else:
                    total_time = 0
                    total_error = 0
                    count = 10

                    for _ in range(count):
                        start = time.time()
                        result = method(A, V, M)
                        end = time.time()
                        total_time += end - start
                        total_error += norm(result - true_values[N])

                    comp_time = total_time / count
                    error = total_error / count

                # Add row to data
                data = pd.concat(
                    [data, pd.DataFrame({
                        "N": [N],
                        "M": [M],
                        "Method": [method_name],
                        "Tau": [tau],
                        "Error": [error],
                        "Computation Time": [comp_time]
                    })],
                    ignore_index=True
                )

# Save results to CSV
data.to_csv("Experiment_Data.csv", index=False)
print("Experiment completed and saved to Experiment_Data.csv")
