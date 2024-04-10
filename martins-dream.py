import functools
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import random
import pandas as pd
import operator

from enum import Enum
from multiprocessing import Pool
from typing import *

class Strategy(Enum):
  AlwaysCooperate = 'ALLC'
  TitForTat = 'TFT'
  AlwaysDefect = 'ALLD'

STRATEGIES = [Strategy.AlwaysCooperate, Strategy.TitForTat, Strategy.AlwaysDefect]

B = 5
C = 1

# (I play as .., they play as ..) -> my payoff
Payoffs = Dict[Tuple[Strategy, Strategy], float]

M: Payoffs = {
  (Strategy.TitForTat, Strategy.TitForTat): (B-C)/2,
  (Strategy.TitForTat, Strategy.AlwaysCooperate): B-C,
  (Strategy.TitForTat, Strategy.AlwaysDefect): 0,
  (Strategy.AlwaysCooperate, Strategy.TitForTat): B-C,
  (Strategy.AlwaysCooperate, Strategy.AlwaysCooperate): B-C,
  (Strategy.AlwaysCooperate, Strategy.AlwaysDefect): -C,
  (Strategy.AlwaysDefect, Strategy.TitForTat): 0,
  (Strategy.AlwaysDefect, Strategy.AlwaysCooperate): B,
  (Strategy.AlwaysDefect, Strategy.AlwaysDefect): 0,
}

def calculate_payoffs(N: int, nALLC: int, nTFT: int, M: Payoffs):
  nALLD = N-nALLC-nTFT
  payoff_ALLC = (
    nALLD/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.AlwaysDefect)]
    + nTFT/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.TitForTat)]
    + (nALLC-1)/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.AlwaysCooperate)]
  )
  payoff_TFT = (
    nALLD/(N-1) * M[(Strategy.TitForTat, Strategy.AlwaysDefect)]
    + (nTFT-1)/(N-1) * M[(Strategy.TitForTat, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.TitForTat, Strategy.AlwaysCooperate)]
  )
  payoff_ALLD = (
    (nALLD-1)/(N-1) * M[(Strategy.AlwaysDefect, Strategy.AlwaysDefect)]
    + nTFT/(N-1) * M[(Strategy.AlwaysDefect, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.AlwaysDefect, Strategy.AlwaysCooperate)]
  )

  return (payoff_ALLC, payoff_TFT, payoff_ALLD)


def pick_individual_with_payoffs(N: int, nALLC: int, nTFT: int, M: Payoffs):
  calculate_payoffs(N, nALLC, nTFT, M, np.exp)
  unnormalized_fitnesses = np.array(list(map(np.exp, calculate_payoffs(N, nALLC, nTFT, M))))
  total_fitness = sum(unnormalized_fitnesses)
  return random.choices(population=STRATEGIES, weights=unnormalized_fitnesses / total_fitness)[0]


def pick_individual_uniformly(N: int, nALLC: int, nTFT: int, nALLD):
  return random.choices(population=STRATEGIES, weights=np.array([nALLC, nTFT, nALLD])/N)[0]

def pairwise_comparison(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_picked_for_update = pick_individual_uniformly(N, nALLC, nTFT, nALLD)
  strategy_role_model = pick_individual_uniformly(
    N-1,
    nALLC-int(strategy_picked_for_update==Strategy.AlwaysCooperate),
    nTFT-int(strategy_picked_for_update==Strategy.TitForTat),
    M,
  )
  ps = {s: pi for s, pi in zip(STRATEGIES, calculate_payoffs(N, nALLC, nTFT, M))}
  ns = {s: nX for s, nX in zip(STRATEGIES, (nALLC, nTFT, nALLD))}

  F_j = ps[strategy_picked_for_update] / ns[strategy_picked_for_update]
  F_i = ps[strategy_role_model] / ns[strategy_role_model]
  x = F_i - F_j
  theta = 1/(1+np.exp(-x)) 
  return random.choices([strategy_role_model, strategy_picked_for_update], weights=[theta, 1-theta])[0]


def imitation(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_picked_for_update = pick_individual_uniformly(N, nALLC, nTFT, nALLD)
  strategy_role_model = pick_individual_with_payoffs(N, nALLC, nTFT, M)

  # Conditional mutation.
  new_strategy = strategy_role_model
  if new_strategy == Strategy.AlwaysCooperate and nALLD > 0:
    new_strategy = random.choices(population=[Strategy.TitForTat, new_strategy], weights=[mu, 1-mu])[0]
    
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )

def simulate(N: int, TRIALS: int, mu: float):
  print('start', mu)
  fixated_ALLD = []
  fixated_ALLC_given_ALLD_extinct = []
  for _ in range(TRIALS):
    nALLC = N-1
    nTFT = 0
    nALLD = 1
    while all(nX < N for nX in (nALLC, nTFT, nALLD)):
      # print(nALLC, nTFT, nALLD)
      nALLCp, nTFTp = imitation(N, nALLC, nTFT, M, mu=mu)
      nALLC, nTFT, nALLD = (nALLCp, nTFTp, N-(nALLCp + nTFTp))

    # print(nALLC, nTFT, nALLD)
    fixated_ALLD.append(nALLD == N)
    if nALLD == 0:
      fixated_ALLC_given_ALLD_extinct.append(nALLC == N)


  fp_ALLD = np.mean(fixated_ALLD)
  fp_ALLC_given_ALLD_extinct = np.mean(fixated_ALLC_given_ALLD_extinct)
  print('end', mu)
  return (N, mu, fp_ALLD, fp_ALLC_given_ALLD_extinct)

def main():
  INTERVALS = 100
  TRIALS = 1_000
  NUM_WORKERS = 8
  data = []
  for N in (10,):
    mus = np.linspace(0, 1, INTERVALS, endpoint=True)
    with Pool(NUM_WORKERS) as p:
      for datum in p.map(functools.partial(simulate, N, TRIALS), mus):
        data.append(datum)

  df = pd.DataFrame(data, columns=['N', 'mu', 'fp_ALLD', 'fp_ALLC_given_ALLD_extinct'])
  fig, ax = plt.subplots(1,2)
  sns.lineplot(df, ax=ax[0], x='mu', y='fp_ALLD', hue='N', linestyle='--', marker='o', legend=False)
  sns.lineplot(df, ax=ax[1], x='mu', y='fp_ALLC_given_ALLD_extinct', hue='N', linestyle='--', marker='o', legend=False)
  plt.tight_layout()
  fig.savefig(f'figs/M-fp_ALLD-saptarshi-{TRIALS=}.png', dpi=300)
  plt.show()

if __name__ == '__main__':
  main()