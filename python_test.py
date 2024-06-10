import linear_algebra as la
import numpy as np
import time

# 2 random matrices
a = np.random.rand(500, 500)
b = np.random.rand(500, 500)

# Matrix multiplication

start = time.time()
c = la.mul_matrices(a, b)
end = time.time()
rust_impl_time = end - start
print('Rust implementation done')

# Numpy matrix multiplication
start = time.time()
c = np.matmul(a, b)
end = time.time()
numpy_impl_time = end - start
print('Numpy done')

# Native matrix multiplication
def matrix_mul(a, b):
    n = len(a)
    c = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c
start = time.time()
c = matrix_mul(a, b)
end = time.time()
native_impl_time = end - start
print('Native python done')

# Table with results
print("Implementation\tTime\tSpeedup")
print(f"Rust\t{rust_impl_time:.4f}\t1.0")
print(f"Numpy\t{numpy_impl_time:.4f}\t{numpy_impl_time / rust_impl_time:.4f}")
print(f"Native\t{native_impl_time:.4f}\t{native_impl_time / rust_impl_time:.4f}")




