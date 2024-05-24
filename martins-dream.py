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
  ) if nALLC > 0 else -np.inf 
  payoff_TFT = (
    nALLD/(N-1) * M[(Strategy.TitForTat, Strategy.AlwaysDefect)]
    + (nTFT-1)/(N-1) * M[(Strategy.TitForTat, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.TitForTat, Strategy.AlwaysCooperate)]
  ) if nTFT > 0 else -np.inf
  payoff_ALLD = (
    (nALLD-1)/(N-1) * M[(Strategy.AlwaysDefect, Strategy.AlwaysDefect)]
    + nTFT/(N-1) * M[(Strategy.AlwaysDefect, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.AlwaysDefect, Strategy.AlwaysCooperate)]
  ) if nALLD > 0 else -np.inf

  return (payoff_ALLC, payoff_TFT, payoff_ALLD)

def pick_individual_with_payoffs(N: int, nALLC: int, nTFT: int, M: Payoffs):
  unnormalized_fitnesses = (
    np.array((nALLC, nTFT, N-nALLC-nTFT)) *
    np.array(list(map(np.exp, calculate_payoffs(N, nALLC, nTFT, M))))
  )
  return random.choices(population=STRATEGIES, weights=unnormalized_fitnesses)[0]

def pick_individual_uniformly(N: int, nALLC: int, nTFT: int, nALLD):
  return random.choices(population=STRATEGIES, weights=np.array([nALLC, nTFT, nALLD]))[0]

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
    nALLD-int(strategy_picked_for_update==Strategy.AlwaysDefect),
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
  # print(nALLC, nTFT, nALLD, strategy_picked_for_update, strategy_role_model)

  # Conditional mutation.
  new_strategy = possibly_mutate(strategy_role_model, N, nTFT, nALLD, mu, back_mu)
    
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )

def death_birth(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_to_die = pick_individual_uniformly(N, nALLC, nTFT, nALLD)
  strategy_to_give_birth = pick_individual_with_payoffs(
    N-1,
    nALLC-int(strategy_to_die==Strategy.AlwaysCooperate),
    nTFT-int(strategy_to_die==Strategy.TitForTat),
    M,
  )
  new_strategy = possibly_mutate(strategy_to_give_birth, N, nTFT, nALLD, mu, back_mu)
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_to_die == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_to_die == Strategy.TitForTat),
  )

def birth_death(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_to_give_birth = pick_individual_with_payoffs(N, nALLC, nTFT, M)
  strategy_to_die = pick_individual_uniformly(
    N-1,
    nALLC-int(strategy_to_give_birth==Strategy.AlwaysCooperate),
    nTFT-int(strategy_to_give_birth==Strategy.TitForTat),
    nALLD-int(strategy_to_give_birth==Strategy.AlwaysDefect),
  )
  new_strategy = possibly_mutate(strategy_to_give_birth, N, nTFT, nALLD, mu, back_mu)
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_to_die == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_to_die == Strategy.TitForTat),
  )

# import networkx as nx
# 
# def birth_death_graph(G: nx.DiGraph, individuals: Dict[Any, Strategy], M: Payoffs, mu: float, back_mu: float):
#   assert 0 <= mu <= 1, mu
#   assert 0 <= back_mu <= 1, back_mu
#   assert len(individuals) == len(G)
# 
# 
#   strategy_to_give_birth = pick_individual_with_payoffs(N, nALLC, nTFT, M)
#   strategy_to_die = pick_individual_uniformly(
#     N-1,
#     nALLC-int(strategy_to_give_birth==Strategy.AlwaysCooperate),
#     nTFT-int(strategy_to_give_birth==Strategy.TitForTat),
#     nALLD-int(strategy_to_give_birth==Strategy.AlwaysDefect),
#   )
#   new_strategy = possibly_mutate(strategy_to_give_birth, N, nTFT, nALLD, mu, back_mu)
#   return (
#     nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_to_die == Strategy.AlwaysCooperate),
#     nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_to_die == Strategy.TitForTat),
#   )


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


from collections import defaultdict

def simulate(N: int, TRIALS: int, dynamics, mu: float, back_mu: float):
  print('start', mu, back_mu)
  fixated_ALLD = []
  fixated = defaultdict(list)
  fixated_ALLC_given_ALLD_extinct = []
  fractions_ALLC_when_ALLD_extinction = []
  for trial in range(TRIALS):
    nALLC = N-1
    nTFT = 0
    nALLD = 1
    tracked_ALLD_extinction = False
    # print(f'trial #{trial}:')
    while all(nX < N for nX in (nALLC, nTFT, nALLD)):
      # print(f'{nALLC=}, {nTFT=}, {nALLD=}')
      # input()
      nALLCp, nTFTp = dynamics(N, nALLC, nTFT, M, mu=mu, back_mu=back_mu)
      nALLC, nTFT, nALLD = (nALLCp, nTFTp, N-(nALLCp + nTFTp))
      if nALLD == 0 and not tracked_ALLD_extinction:
        fractions_ALLC_when_ALLD_extinction.append(nALLC / N)
        tracked_ALLD_extinction = True
    # print(f'{nALLC=}, {nTFT=}, {nALLD=}')
    # input()

    fixated_ALLD.append(nALLD == N)
    for strategy, nStrategy in zip(STRATEGIES, (nALLC, nTFT, nALLD)):
      fixated[strategy].append(nStrategy == N)
    if nALLD == 0:
      fixated_ALLC_given_ALLD_extinct.append(nALLC == N)


  fp_ALLD = np.mean(fixated_ALLD)
  fp: Dict[Strategy, float] = {}
  for strategy in STRATEGIES:
    fp[strategy] = np.mean(fixated[strategy])
  fp_ALLC_given_ALLD_extinct = np.mean(fixated_ALLC_given_ALLD_extinct)
  fraction_ALLC_when_ALLD_extinction = np.mean(fractions_ALLC_when_ALLD_extinction)
  print('end', mu, back_mu)
  return (N, mu, back_mu, *(fp[strategy] for strategy in STRATEGIES), fp_ALLC_given_ALLD_extinct, fraction_ALLC_when_ALLD_extinction)


DYNAMICS = {
  'imitation': imitation,
  'pairwise-comparison': pairwise_comparison,
  'birth-death': birth_death,
  'death-birth': death_birth,
  'wrong': wrong,
}


import itertools
INTERVALS = 10
TRIALS = 1000
NUM_WORKERS = 1
DYNAMIC = 'birth-death'
N = 10

def collect_data():
  data = []
  TICKS = np.linspace(0, 1, INTERVALS, endpoint=True)
  TICK_LABELS = [('0' if tick == 0 else '1' if tick == 1 else '') for tick in TICKS]
  mus = TICKS
  # back_mus = TICKS
  mutations = [(mu, 0) for mu in mus]
  # mutations = list(itertools.product(mus, back_mus))
  # print(mutations)
  if NUM_WORKERS > 1:
    with Pool(NUM_WORKERS) as p:
      for datum in p.starmap(functools.partial(simulate, N, TRIALS, DYNAMICS[DYNAMIC]), mutations):
        data.append(datum)
  else:
    for datum in itertools.starmap(functools.partial(simulate, N, TRIALS, DYNAMICS[DYNAMIC]), mutations):
      data.append(datum)

  return pd.DataFrame(data, columns=['N', 'mu', 'back_mu', 'fp_ALLC', 'fp_TFT', 'fp_ALLD', 'fp_ALLC_given_ALLD_extinct', 'fraction_ALLC_when_ALLD_extinct'])

def stack_plot(df: pd.DataFrame):
  # df = df[['fp_ALLD', 'fp_ALLC_given_ALLD_extinct', 'mu']]
  # df['fp_ALLC'] = df['fp_ALLC_given_ALLD_extinct'] * (1-df['fp_ALLD'])
  # df['fp_TFT'] = 1-(df['fp_ALLC'] + df['fp_ALLD'])
  df = df.drop(columns=['fp_ALLC_given_ALLD_extinct'])
  df = df[['mu', 'fp_ALLD', 'fp_TFT', 'fp_ALLC']]
  df = df.rename(columns={'fp_ALLD': 'ALLD', 'fp_TFT': 'TFT', 'fp_ALLC': 'ALLC'})
  ax = df.set_index('mu').plot(kind='area')
  ax.set_ylabel(r'Fixation probability, $p$')
  ax.set_xlabel(r'Mutation rate, $\mu$')
  plt.legend(loc='upper right')
  fig = ax.get_figure()
  fig.suptitle(f"{DYNAMIC=}, {N=}, {TRIALS=}, {INTERVALS=}")
  fig.savefig(get_plot_file_name(), dpi=300)
  plt.show()

def plot(df: pd.DataFrame): ...
  # print(df)
  # fig, ax = plt.subplots(1,3)
  # for i, value in enumerate(('fp_ALLD', 'fp_ALLC_given_ALLD_extinct', 'fraction_ALLC_when_ALLD_extinct')):
    # sns.heatmap(
    #   data=df.pivot(index='back_mu', columns='mu', values=value).sort_index(ascending=False, level=0),
    #   ax=ax[i],
    #   xticklabels=TICK_LABELS,
    #   yticklabels=list(reversed(TICK_LABELS)),
    #   vmin=0,
    #   vmax=1,
    #   cbar_kws={'label': value, "shrink": 0.25}
    # ) 
    # ax[i].axis('scaled')
    # sns.lineplot(df, ax=ax[i], x='mu', y=value, hue='N', linestyle='--', marker='o', legend=False)
    # ax[i].set_ylim((0, 1))
  # sns.lineplot(df, ax=ax[2], x='mu', y='fraction_ALLC_when_ALLD_extinct', hue='N', linestyle='--', marker='o', legend=False)
  # for i in range(3):

  # plt.tight_layout()
  # fig.suptitle(f"{DYNAMIC=}, {N=}, {TRIALS=}")
  # fig.savefig(f'figs/saptarshi-custom.png', dpi=300)
  # plt.show()

def get_file_name() -> str:
  return f'DYNAMIC:{DYNAMIC}-N{N}-TRIALS{TRIALS}-INTERVALS{INTERVALS}'

def get_plot_file_name() -> str:
  return f'figs/{get_file_name()}.png'

def get_data_file_name() -> str:
  return f'data/{get_file_name()}.json'

from pathlib import Path

def load_data():
  with Path(get_data_file_name()).open('r') as f:
    return pd.read_json(f, orient='records')

def store_data(df: pd.DataFrame):
  with Path(get_data_file_name()).open('w') as f:
    df.to_json(f, orient='records')

USE_EXISTING_DATA = False
def main():
  df = (load_data if USE_EXISTING_DATA else collect_data)()
  store_data(df)
  stack_plot(df)

if __name__ == '__main__':
  main()