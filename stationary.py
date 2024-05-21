import numpy as np
from sympy import *


def main():
  mu1 = .1
  mu2 = .3
  N = 3
  T = np.zeros(shape=(N+1,N+1))
  for k in range(N+1):
    acc = 0
    if k-1 >= 0:
      T[k,k-1] = k/N*mu2
      acc += T[k,k-1]
    if k+1 <= N:
      T[k,k+1] = (1-k/N)*mu1
      acc += T[k,k+1]
    T[k,k] = 1-acc

  # print(T)
  u = np.ones(shape=(N+1,)) @ np.linalg.inv(np.identity(N+1) + np.ones(shape=(N+1,N+1)) - T)
  # x = np.zeros(shape=(N+1,))
  # x[1] = 1
  # up = np.linalg.matrix_power(T, 100)@x
  # up = up / np.sum(up)
  print(u)
  print(sum(k*u[k] for k in range(N+1)), mu2/(mu1+mu2))
  # print(up)


if __name__ == '__main__':
  main()