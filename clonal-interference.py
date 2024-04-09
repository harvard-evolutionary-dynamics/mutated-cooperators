import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import random
import pandas as pd

from enum import Enum
from typing import *

class Individual(Enum):
  Cooperator = 'C'
  MutatedCooperator = 'C1'
  Defector = 'D'

INDIVIDUALS = [Individual.Cooperator, Individual.MutatedCooperator, Individual.Defector]

# (I play as .., they play as ..) -> my payoff
Payoffs = Dict[Tuple[Individual, Individual], float]

M1: Payoffs = {
  (Individual.MutatedCooperator, Individual.MutatedCooperator): 1,
  (Individual.MutatedCooperator, Individual.Cooperator): 2,
  (Individual.MutatedCooperator, Individual.Defector): 0,
  (Individual.Cooperator, Individual.MutatedCooperator): 2,
  (Individual.Cooperator, Individual.Cooperator): 4,
  (Individual.Cooperator, Individual.Defector): 1,
  (Individual.Defector, Individual.MutatedCooperator): 3,
  (Individual.Defector, Individual.Cooperator): 6,
  (Individual.Defector, Individual.Defector): 2,
}

M2: Payoffs = {
  (Individual.MutatedCooperator, Individual.MutatedCooperator): 2.5,
  (Individual.MutatedCooperator, Individual.Cooperator): 5,
  (Individual.MutatedCooperator, Individual.Defector): 1.5,
  (Individual.Cooperator, Individual.MutatedCooperator): 2,
  (Individual.Cooperator, Individual.Cooperator): 4,
  (Individual.Cooperator, Individual.Defector): 1,
  (Individual.Defector, Individual.MutatedCooperator): 3,
  (Individual.Defector, Individual.Cooperator): 6,
  (Individual.Defector, Individual.Defector): 2,
}

M3: Payoffs = {
  (Individual.MutatedCooperator, Individual.MutatedCooperator): 4,
  (Individual.MutatedCooperator, Individual.Cooperator): 2,
  (Individual.MutatedCooperator, Individual.Defector): 0,
  (Individual.Cooperator, Individual.MutatedCooperator): 2,
  (Individual.Cooperator, Individual.Cooperator): 4,
  (Individual.Cooperator, Individual.Defector): 1,
  (Individual.Defector, Individual.MutatedCooperator): 1,
  (Individual.Defector, Individual.Cooperator): 6,
  (Individual.Defector, Individual.Defector): 2,
}

M4: Payoffs = {
  (Individual.MutatedCooperator, Individual.MutatedCooperator): 4,
  (Individual.MutatedCooperator, Individual.Cooperator): 2,
  (Individual.MutatedCooperator, Individual.Defector): 0,
  (Individual.Cooperator, Individual.MutatedCooperator): 1,
  (Individual.Cooperator, Individual.Cooperator): 4,
  (Individual.Cooperator, Individual.Defector): 1,
  (Individual.Defector, Individual.MutatedCooperator): 2,
  (Individual.Defector, Individual.Cooperator): 6,
  (Individual.Defector, Individual.Defector): 2,
}

def birth_death(N: int, nC: int, nC1: int, M: Payoffs, mu: float = 0, w: float = 1):
  nD = N-nC-nC1
  assert nC + nC1 + nD == N
  assert all(nX >= 0 for nX in (nC, nC1, nD))
  payoff_C = (
    nD/(N-1) * M[(Individual.Cooperator, Individual.Defector)]
    + nC1/(N-1) * M[(Individual.Cooperator, Individual.MutatedCooperator)]
    + (nC-1)/(N-1) * M[(Individual.Cooperator, Individual.Cooperator)]
  )
  payoff_C1 = (
    nD/(N-1) * M[(Individual.MutatedCooperator, Individual.Defector)]
    + (nC1-1)/(N-1) * M[(Individual.MutatedCooperator, Individual.MutatedCooperator)]
    + nC/(N-1) * M[(Individual.MutatedCooperator, Individual.Cooperator)]
  )
  payoff_D = (
    (nD-1)/(N-1) * M[(Individual.Defector, Individual.Defector)]
    + nC1/(N-1) * M[(Individual.Defector, Individual.MutatedCooperator)]
    + nC/(N-1) * M[(Individual.Defector, Individual.Cooperator)]
  )

  f_C = 1 - w*(1-payoff_C)
  f_C1 = 1 - w*(1-payoff_C1)
  f_D = 1 - w*(1-payoff_D)

  total_f = nC*f_C + nC1*f_C1 + nD*f_D

  (d_nC, d_nC1, d_nD), = random.choices(*zip(*[
    ((1-1,0,0), (nC*f_C / total_f) * nC/N * (1-mu)), # C <- C 
    ((1,-1,0), (nC*f_C / total_f) * nC1/N * (1-mu)), # C <- C1
    ((1,0,-1), (nC*f_C / total_f) * nD/N * (1-mu)),  # C <- D
    ((-1,1,0), (nC1*f_C1 / total_f) * nC/N + (nC*f_C / total_f) * nC/N * mu), # C1 <- C
    ((0,1-1,0), (nC1*f_C1 / total_f) * nC1/N + (nC*f_C / total_f) * nC1/N * mu), # C1 <- C1
    ((0,1,-1), (nC1*f_C1 / total_f) * nD/N + (nC*f_C / total_f) * nD/N * mu),   # C1 <- D
    ((-1,0,1), (nD*f_D / total_f) * nC/N),     # D <- C
    ((0,-1,1), (nD*f_D / total_f) * nC1/N),   # D <- C1
    ((0,0,1-1), (nD*f_D / total_f) * nD/N),   # D <- D
  ]))
  
  nCp = nC+d_nC
  nC1p = nC1+d_nC1
  nDp = nD+d_nD

  return (nCp, nC1p)

def main():
  INTERVALS = 6
  # epsilon, delta = 0.001, .05
  # TRIALS = int(np.ceil(4*np.log(2/delta)/epsilon**2)) # 10000
  # print(f"{epsilon=}, {delta=} --> {TRIALS=}")
  TRIALS = 10_000
  w = 1
  M = M4
  data = []
  for N in range(10, 50+1, 10):
    for mu in np.linspace(0, 0.5, INTERVALS, endpoint=True):
      fixated_D = []
      stepss = []
      for trial in range(TRIALS):
        nC = N-1
        nC1 = 0
        nD = 1
        steps = 0
        while nD not in (0, N):
          nCp, nC1p = birth_death(N, nC, nC1, M, mu=mu, w=w)
          nC, nC1, nD = (nCp, nC1p, N-(nCp + nC1p))
          steps += 1
        fixated_D.append(nD == N)
        if nD == N:
          stepss.append(steps)

      fp_D = np.mean(fixated_D)
      ft_D = np.mean(stepss)
      data.append((N, mu, fp_D, ft_D))

  df = pd.DataFrame(data, columns=['N', 'mu', 'fp_D', 'ft_D'])
  # sns.lineplot(df, x='mu', y='fp_D', hue='N', linestyle='--', marker='o')
  sns.lineplot(df, x='mu', y='ft_D', hue='N', linestyle='--', marker='o')
  plt.savefig(f'figs/M4-ftD-saptarshi.png', dpi=300, bbox_inches="tight")
  plt.show()

if __name__ == '__main__':
  main()