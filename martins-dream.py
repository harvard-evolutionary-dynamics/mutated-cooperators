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
  calculate_payoffs(N, nALLC, nTFT, M)
  unnormalized_fitnesses = np.array(list(map(np.exp, calculate_payoffs(N, nALLC, nTFT, M))))
  total_fitness = sum(unnormalized_fitnesses)
  return random.choices(population=STRATEGIES, weights=unnormalized_fitnesses / total_fitness)[0]


def pick_individual_uniformly(N: int, nALLC: int, nTFT: int, nALLD):
  return random.choices(population=STRATEGIES, weights=np.array([nALLC, nTFT, nALLD])/N)[0]

def possibly_mutate(strategy: Strategy, N: int, nTFT: int, nALLD: int, mu: float, back_mu: float):
  if strategy == Strategy.AlwaysCooperate and nALLD > 0:
    strategy = random.choices(population=[Strategy.TitForTat, strategy], weights=[mu, 1-mu])[0]
  if strategy == Strategy.TitForTat and nALLD == 0 and nTFT < N:
    strategy = random.choices(population=[Strategy.AlwaysCooperate, strategy], weights=[back_mu, 1-back_mu])[0]

  return strategy

def pairwise_comparison(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  # print(N, nALLC, nTFT, M, mu, back_mu)
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_picked_for_update = pick_individual_uniformly(N, nALLC, nTFT, nALLD)
  strategy_role_model = pick_individual_uniformly(
    N-1,
    nALLC-int(strategy_picked_for_update==Strategy.AlwaysCooperate),
    nTFT-int(strategy_picked_for_update==Strategy.TitForTat),
    nALLD,
  )
  ps = {s: pi for s, pi in zip(STRATEGIES, calculate_payoffs(N, nALLC, nTFT, M))}

  F_j = ps[strategy_picked_for_update]
  F_i = ps[strategy_role_model]
  x = F_i - F_j
  theta = 1/(1+np.exp(-x)) 
  assert 0 <= theta <= 1, theta

  new_strategy = strategy_picked_for_update
  if random.random() < theta:
    new_strategy = possibly_mutate(strategy_role_model, N, nTFT, nALLD, mu, back_mu)

  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )
    

def imitation(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_picked_for_update = pick_individual_uniformly(N, nALLC, nTFT, nALLD)
  strategy_role_model = pick_individual_with_payoffs(N, nALLC, nTFT, M)

  # Conditional mutation.
  new_strategy = possibly_mutate(strategy_role_model, N, nTFT, nALLD, mu, back_mu)
    
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )

def simulate(N: int, TRIALS: int, dynamics, mu: float, back_mu: float):
  print('start', mu, back_mu)
  fixated_ALLD = []
  fixated_ALLC_given_ALLD_extinct = []
  fractions_ALLC_when_ALLD_extinction = []
  for _ in range(TRIALS):
    nALLC = N-1
    nTFT = 0
    nALLD = 1
    tracked_ALLD_extinction = False
    while all(nX < N for nX in (nALLC, nTFT, nALLD)):
      nALLCp, nTFTp = dynamics(N, nALLC, nTFT, M, mu=mu, back_mu=back_mu)
      nALLC, nTFT, nALLD = (nALLCp, nTFTp, N-(nALLCp + nTFTp))
      if nALLD == 0 and not tracked_ALLD_extinction:
        fractions_ALLC_when_ALLD_extinction.append(nALLC / N)
        tracked_ALLD_extinction = True

    fixated_ALLD.append(nALLD == N)
    if nALLD == 0:
      fixated_ALLC_given_ALLD_extinct.append(nALLC == N)

  fp_ALLD = np.mean(fixated_ALLD)
  fp_ALLC_given_ALLD_extinct = np.mean(fixated_ALLC_given_ALLD_extinct)
  fraction_ALLC_when_ALLD_extinction = np.mean(fractions_ALLC_when_ALLD_extinction)
  print('end', mu, back_mu)
  return (N, mu, fp_ALLD, fp_ALLC_given_ALLD_extinct, fraction_ALLC_when_ALLD_extinction)

def wrong(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)
  pick_individual = lambda: random.choices(population=STRATEGIES, weights=np.array([nALLC, nTFT, nALLD])/N)[0]
  strategy_picked_for_update = pick_individual()
  strategy_role_model = pick_individual()

  # payoff_picked_for_update = M[strategy_picked_for_update, strategy_picked_for_update]
  payoff_picked_for_update = M[strategy_picked_for_update, strategy_role_model]
  payoff_role_model = M[strategy_role_model, strategy_picked_for_update]

  F = lambda x: np.exp(x)
  fitnesses = np.array(list(map(F, (payoff_picked_for_update, payoff_role_model))))
  total_fitness = sum(fitnesses)
  winning_strategy = random.choices(population=[strategy_picked_for_update, strategy_role_model], weights=fitnesses/total_fitness)[0]

  # Conditional mutation.
  new_strategy = possibly_mutate(winning_strategy, N, nTFT, nALLD, mu, back_mu)

  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )

DYNAMICS = {
  'imitation': imitation,
  'pairwise_comparison': pairwise_comparison,
  'wrong': wrong,
}

import itertools

def main():
  INTERVALS = 20
  TRIALS = 10_000
  NUM_WORKERS = 8
  data = []
  for N in (10,):
    mus = np.linspace(0, 1, INTERVALS, endpoint=True)
    # back_mus = np.linspace(0, 1, INTERVALS, endpoint=True)
    mutations = [(mu, 0) for mu in mus]# itertools.product(mus, back_mus)
    print(mutations)
    with Pool(NUM_WORKERS) as p:
      for datum in p.starmap(functools.partial(simulate, N, TRIALS, DYNAMICS['pairwise_comparison']), mutations):
        data.append(datum)

  df = pd.DataFrame(data, columns=['N', 'mu', 'fp_ALLD', 'fp_ALLC_given_ALLD_extinct', 'fraction_ALLC_when_ALLD_extinct'])

  fig, ax = plt.subplots(1,3)
  sns.lineplot(df, ax=ax[0], x='mu', y='fp_ALLD', hue='N', linestyle='--', marker='o', legend=False)
  sns.lineplot(df, ax=ax[1], x='mu', y='fp_ALLC_given_ALLD_extinct', hue='N', linestyle='--', marker='o', legend=False)
  sns.lineplot(df, ax=ax[2], x='mu', y='fraction_ALLC_when_ALLD_extinct', hue='N', linestyle='--', marker='o', legend=False)
  for i in range(3):
    ax[i].set_ylim((0, 1))

  plt.tight_layout()
  fig.savefig(f'figs/M-fp_ALLD-saptarshi-{TRIALS=}-pc-fraction-back-mu.png', dpi=300)
  plt.show()

if __name__ == '__main__':
  main()