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
from collections import Counter
from customlogger import logger

class Strategy(Enum):
  AlwaysCooperate = 'ALLC'
  TitForTat = 'TFT'
  AlwaysDefect = 'ALLD'
  WinStayLoseShift = 'WSLS'

STRATEGIES = [Strategy.AlwaysCooperate, Strategy.TitForTat, Strategy.WinStayLoseShift, Strategy.AlwaysDefect]

B = 5
C = 1

# (I play as .., they play as ..) -> my payoff
Payoffs = Dict[Tuple[Strategy, Strategy], float]

M: Payoffs = {
  (Strategy.TitForTat, Strategy.TitForTat): (B-C)/2,
  (Strategy.TitForTat, Strategy.AlwaysCooperate): B-C,
  (Strategy.TitForTat, Strategy.AlwaysDefect): 0,
  (Strategy.TitForTat, Strategy.WinStayLoseShift): (B-C)/2,
  (Strategy.AlwaysCooperate, Strategy.TitForTat): B-C,
  (Strategy.AlwaysCooperate, Strategy.AlwaysCooperate): B-C,
  (Strategy.AlwaysCooperate, Strategy.AlwaysDefect): -C,
  (Strategy.AlwaysCooperate, Strategy.WinStayLoseShift): (B-2*C)/2,
  (Strategy.AlwaysDefect, Strategy.TitForTat): 0,
  (Strategy.AlwaysDefect, Strategy.AlwaysCooperate): B,
  (Strategy.AlwaysDefect, Strategy.AlwaysDefect): 0,
  (Strategy.AlwaysDefect, Strategy.WinStayLoseShift): B/2,
  (Strategy.WinStayLoseShift, Strategy.TitForTat): (B-C)/2,
  (Strategy.WinStayLoseShift, Strategy.AlwaysCooperate): (2*B-C)/2,
  (Strategy.WinStayLoseShift, Strategy.AlwaysDefect): -C/2,
  (Strategy.WinStayLoseShift, Strategy.WinStayLoseShift): B-C,
}

def calculate_payoffs(N: int, nALLC: int, nTFT: int, nWSLS: int, nALLD: int, M: Payoffs):
  payoff_ALLC = (
    nWSLS/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.WinStayLoseShift)]
    + nALLD/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.AlwaysDefect)]
    + nTFT/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.TitForTat)]
    + (nALLC-1)/(N-1) * M[(Strategy.AlwaysCooperate, Strategy.AlwaysCooperate)]
  ) if nALLC > 0 else -np.inf 
  payoff_TFT = (
    nWSLS/(N-1) * M[(Strategy.TitForTat, Strategy.WinStayLoseShift)]
    + nALLD/(N-1) * M[(Strategy.TitForTat, Strategy.AlwaysDefect)]
    + (nTFT-1)/(N-1) * M[(Strategy.TitForTat, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.TitForTat, Strategy.AlwaysCooperate)]
  ) if nTFT > 0 else -np.inf
  payoff_ALLD = (
    nWSLS/(N-1) * M[(Strategy.AlwaysDefect, Strategy.WinStayLoseShift)]
    + (nALLD-1)/(N-1) * M[(Strategy.AlwaysDefect, Strategy.AlwaysDefect)]
    + nTFT/(N-1) * M[(Strategy.AlwaysDefect, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.AlwaysDefect, Strategy.AlwaysCooperate)]
  ) if nALLD > 0 else -np.inf
  payoff_WSLS = (
    (nWSLS-1)/(N-1) * M[(Strategy.WinStayLoseShift, Strategy.WinStayLoseShift)]
    + nALLD/(N-1) * M[(Strategy.WinStayLoseShift, Strategy.AlwaysDefect)]
    + nTFT/(N-1) * M[(Strategy.WinStayLoseShift, Strategy.TitForTat)]
    + nALLC/(N-1) * M[(Strategy.WinStayLoseShift, Strategy.AlwaysCooperate)]
  ) if nALLD > 0 else -np.inf

  return (payoff_ALLC, payoff_TFT, payoff_WSLS, payoff_ALLD)

def pick_individual_with_payoffs(N: int, nALLC: int, nTFT: int, nWSLS: int, nALLD: int, M: Payoffs):
  unnormalized_fitnesses = (
    np.array((nALLC, nTFT, nWSLS, nALLD)) *
    np.array(list(map(np.exp, calculate_payoffs(N, nALLC, nTFT, nWSLS, nALLD, M))))
  )
  return random.choices(population=STRATEGIES, weights=unnormalized_fitnesses)[0]

def pick_individual_uniformly(N: int, nALLC: int, nTFT: int, nWSLS: int, nALLD: int):
  return random.choices(population=STRATEGIES, weights=np.array([nALLC, nTFT, nWSLS, nALLD]))[0]

def possibly_mutate(strategy: Strategy, N: int, nALLC: int, nTFT: int, nWSLS: int, nALLD: int, mu1: float, mu2: float, back_mu: float):
  if strategy == Strategy.AlwaysCooperate and nALLD > 0:
    return random.choices(population=[Strategy.TitForTat, strategy], weights=[mu1, 1-mu1])[0]
  if strategy == Strategy.AlwaysDefect and nALLC > 0:
    return random.choices(population=[Strategy.WinStayLoseShift, strategy], weights=[mu2, 1-mu2])[0]
  if strategy == Strategy.TitForTat and nALLD == 0 and nTFT < N:
    return random.choices(population=[Strategy.AlwaysCooperate, strategy], weights=[back_mu, 1-back_mu])[0]

  return strategy

def pairwise_comparison(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  raise NotImplementedError()
  # logger.info(N, nALLC, nTFT, M, mu, back_mu)
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
  raise NotImplementedError()
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)

  strategy_picked_for_update = pick_individual_uniformly(N, nALLC, nTFT, nALLD)
  strategy_role_model = pick_individual_with_payoffs(N, nALLC, nTFT, M)
  # logger.info(nALLC, nTFT, nALLD, strategy_picked_for_update, strategy_role_model)

  # Conditional mutation.
  new_strategy = possibly_mutate(strategy_role_model, N, nTFT, nALLD, mu, back_mu)
    
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )

def death_birth(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  raise NotImplementedError()
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

def birth_death(N: int, nALLC: int, nTFT: int, nWSLS: int, nALLD: int, M: Payoffs, mu1: float, mu2: float, back_mu: float):
  assert 0 <= mu1 <= 1, mu1
  assert 0 <= mu2 <= 1, mu2
  assert nALLC + nTFT + + nWSLS + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nWSLS, nALLD)), (nALLC, nTFT, nWSLS, nALLD)

  strategy_to_give_birth = pick_individual_with_payoffs(N, nALLC, nTFT, nWSLS, nALLD, M)
  strategy_to_die = pick_individual_uniformly(
    N-1,
    nALLC-int(strategy_to_give_birth==Strategy.AlwaysCooperate),
    nTFT-int(strategy_to_give_birth==Strategy.TitForTat),
    nWSLS-int(strategy_to_give_birth==Strategy.WinStayLoseShift),
    nALLD-int(strategy_to_give_birth==Strategy.AlwaysDefect),
  )
  new_strategy = possibly_mutate(strategy_to_give_birth, N, nALLC, nTFT, nWSLS, nALLD, mu1, mu2, back_mu)
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_to_die == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_to_die == Strategy.TitForTat),
    nWSLS + int(new_strategy == Strategy.WinStayLoseShift) - int(strategy_to_die == Strategy.WinStayLoseShift),
    nALLD + int(new_strategy == Strategy.AlwaysDefect) - int(strategy_to_die == Strategy.AlwaysDefect),
  )

import networkx as nx

def calculate_payoff_graph(G: nx.DiGraph, strategies: Dict[Any, Strategy], individual: Any, M: Payoffs):
  out_degree = G.out_degree(individual)
  return np.exp(
    sum(
      M[strategies[individual], strategies[neighbor_individual]] / out_degree
      for _, neighbor_individual in G.out_edges(individual)
    )
  )

def birth_death_graph(G_games: nx.DiGraph, G_reproduction: nx.DiGraph, strategies: Dict[Any, Strategy], M: Payoffs, mu: float, back_mu: float):
  assert 0 <= mu <= 1, mu
  assert 0 <= back_mu <= 1, back_mu
  if back_mu > 0:
    raise NotImplementedError('back_mu not incorporated yet')
  assert len(strategies) == len(G_games)
  assert len(G_games) == len(G_reproduction)

  # pick birther.
  payoffs = {
    individual: calculate_payoff_graph(G_games, strategies, individual, M)
    for individual in G_games.nodes()
  }
  individual_to_give_birth = random.choices(
    population=list(G_games.nodes()),
    weights=[payoffs[individual] for individual in G_games.nodes()],
  )[0]

  # pick death location.
  individual_to_die = random.choices([neighbor_individual for _, neighbor_individual in G_reproduction.out_edges(individual_to_give_birth)])[0]

  # possibly mutate.
  new_strategy = strategies[individual_to_give_birth]
  if new_strategy == Strategy.AlwaysCooperate and Strategy.AlwaysDefect in strategies.values():
    new_strategy = random.choices(population=[Strategy.TitForTat, new_strategy], weights=[mu, 1-mu])[0]
  if new_strategy == Strategy.TitForTat and Strategy.AlwaysDefect in strategies.values() and Counter(strategies.values()).get(Strategy.TitForTat, 0) < N:
    new_strategy = random.choices(population=[Strategy.AlwaysCooperate, new_strategy], weights=[back_mu, 1-back_mu])[0]

  strategies[individual_to_die] = new_strategy
  return strategies


def wrong(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float, back_mu: float):
  raise NotImplementedError()
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
import copy

def simulate_graph(G_games: nx.DiGraph, G_reproduction: nx.DiGraph, strategies: Dict[Any, Strategy], TRIALS: int, graph_dynamics, mu: float, back_mu: float):
  N = len(G_games)
  logger.info(('start', mu, back_mu))
  fixated = defaultdict(list)

  for trial in range(TRIALS):
    strategies_p = copy.deepcopy(strategies)
    while len(set(strategies_p.values())) > 1:
      strategies_p = graph_dynamics(G_games, G_reproduction, strategies_p, M, mu=mu, back_mu=back_mu)

    counts = Counter(strategies_p.values())
    for strategy, nStrategy in zip(STRATEGIES, (counts[strategy] for strategy in STRATEGIES)):
      fixated[strategy].append(nStrategy == N)


  fp: Dict[Strategy, float] = {}
  for strategy in STRATEGIES:
    fp[strategy] = np.mean(fixated[strategy])

  logger.info(('end', mu, back_mu))
  return (N, mu, back_mu, *(fp[strategy] for strategy in STRATEGIES))

def simulate(N: int, TRIALS: int, dynamics, mu1: float, mu2: float, back_mu: float):
  logger.info(('start', mu1, mu2, back_mu))
  fixated_ALLD = []
  fixated = defaultdict(list)
  fixated_ALLC_given_ALLD_extinct = []
  fractions_ALLC_when_ALLD_extinction = []
  for trial in range(TRIALS):
    nALLC = N-1
    nTFT = 0
    nWSLS = 0
    nALLD = 1
    tracked_ALLD_extinction = False
    while all(nX < N for nX in (nALLC, nTFT, nWSLS, nALLD)):
      nALLCp, nTFTp, nWSLSp, nALLDp = dynamics(N, nALLC, nTFT, nWSLS, nALLD, M, mu1=mu1, mu2=mu2, back_mu=back_mu)
      nALLC, nTFT, nWSLS, nALLD = (nALLCp, nTFTp, nWSLSp, nALLDp)
      if nALLD == 0 and not tracked_ALLD_extinction:
        fractions_ALLC_when_ALLD_extinction.append(nALLC / N)
        tracked_ALLD_extinction = True

    fixated_ALLD.append(nALLD == N)
    for strategy, nStrategy in zip(STRATEGIES, (nALLC, nTFT, nWSLS, nALLD)):
      fixated[strategy].append(nStrategy == N)
    if nALLD == 0:
      fixated_ALLC_given_ALLD_extinct.append(nALLC == N)


  fp_ALLD = np.mean(fixated_ALLD)
  fp: Dict[Strategy, float] = {}
  for strategy in STRATEGIES:
    fp[strategy] = np.mean(fixated[strategy])
  fp_ALLC_given_ALLD_extinct = np.mean(fixated_ALLC_given_ALLD_extinct)
  fraction_ALLC_when_ALLD_extinction = np.mean(fractions_ALLC_when_ALLD_extinction)
  logger.info(('end', mu1, mu2, back_mu))
  return (N, mu1, mu2, back_mu, *(fp[strategy] for strategy in STRATEGIES), fp_ALLC_given_ALLD_extinct, fraction_ALLC_when_ALLD_extinction)


DYNAMICS = {
  'imitation': imitation,
  'pairwise-comparison': pairwise_comparison,
  'birth-death': birth_death,
  'death-birth': death_birth,
  'birth-death-graph': birth_death_graph,
  'wrong': wrong,
}

import itertools
INTERVALS = 100
TRIALS = 10000
NUM_WORKERS = 8
DYNAMIC = 'birth-death'
N = 10

def collect_data_graph(G_games: nx.DiGraph, G_reproduction: nx.DiGraph, strategies: Dict[Any, Strategy]):
  data = []
  TICKS = np.linspace(0, 1, INTERVALS, endpoint=True)
  TICK_LABELS = [('0' if tick == 0 else '1' if tick == 1 else '') for tick in TICKS]
  mus = TICKS
  # back_mus = TICKS
  mutations = [(mu, 0) for mu in mus]
  # mutations = list(itertools.product(mus, back_mus))
  if NUM_WORKERS > 1:
    with Pool(NUM_WORKERS) as p:
      for datum in p.starmap(functools.partial(simulate_graph, G_games, G_reproduction, strategies, TRIALS, DYNAMICS[DYNAMIC]), mutations):
        data.append(datum)
  else:
    for datum in itertools.starmap(functools.partial(simulate_graph, G_games, G_reproduction, strategies, TRIALS, DYNAMICS[DYNAMIC]), mutations):
      data.append(datum)

  return pd.DataFrame(data, columns=['N', 'mu', 'back_mu', 'fp_ALLC', 'fp_TFT', 'fp_ALLD'])

def collect_data():
  data = []
  TICKS = np.linspace(0, 1, INTERVALS, endpoint=True)
  TICK_LABELS = [('0' if tick == 0 else '1' if tick == 1 else '') for tick in TICKS]
  mus = TICKS
  # back_mus = TICKS
  mutations = [(mu, .5, 0) for mu in mus]
  # mutations = list(itertools.product(mus, back_mus))
  if NUM_WORKERS > 1:
    with Pool(NUM_WORKERS) as p:
      for datum in p.starmap(functools.partial(simulate, N, TRIALS, DYNAMICS[DYNAMIC]), mutations):
        data.append(datum)
  else:
    for datum in itertools.starmap(functools.partial(simulate, N, TRIALS, DYNAMICS[DYNAMIC]), mutations):
      data.append(datum)

  return pd.DataFrame(data, columns=['N', 'mu1', 'mu2', 'back_mu', 'fp_ALLC', 'fp_TFT', 'fp_WSLS', 'fp_ALLD', 'fp_ALLC_given_ALLD_extinct', 'fraction_ALLC_when_ALLD_extinct'])

def stack_plot(dff: pd.DataFrame, **kwargs):
  # df = df[['fp_ALLD', 'fp_ALLC_given_ALLD_extinct', 'mu']]
  # df['fp_ALLC'] = df['fp_ALLC_given_ALLD_extinct'] * (1-df['fp_ALLD'])
  # df['fp_TFT'] = 1-(df['fp_ALLC'] + df['fp_ALLD'])
  df = dff.drop(columns=['fp_ALLC_given_ALLD_extinct'], errors='ignore')
  df = df[['mu1', 'fp_ALLD', 'fp_TFT', 'fp_WSLS', 'fp_ALLC']]
  df = df.rename(columns={'fp_ALLD': 'ALLD', 'fp_TFT': 'TFT', 'fp_ALLC': 'ALLC', 'fp_WSLS': 'WSLS'})
  ax = df.set_index('mu1').plot(kind='area')
  dff[['mu1', 'fraction_ALLC_when_ALLD_extinct']].set_index('mu1').plot.line(ax=ax)
  ax.set_ylabel(r'Fixation probability, $p$')
  ax.set_xlabel(r'Mutation rate, $\mu_1$')
  plt.legend(loc='upper right')
  fig = ax.get_figure()
  fig.suptitle(f"$\\mu_2=0.5$, {DYNAMIC=}, {N=}, {TRIALS=}, {INTERVALS=}")
  fig.savefig(get_plot_file_name(**kwargs), dpi=300)
  plt.show()

def stringify_kwargs(**kwargs):
  return '-'.join(f'{k}:{v}' for k, v in sorted(kwargs.items()))

def get_file_name(**kwargs) -> str:
  x = '-' if kwargs else ''
  return f'WSLS-DYNAMIC:{DYNAMIC}-N{N}-TRIALS{TRIALS}-INTERVALS{INTERVALS}{x}{stringify_kwargs(**kwargs)}'

def get_plot_file_name(**kwargs) -> str:
  return f'figs/{get_file_name(**kwargs)}.png'

def get_data_file_name(**kwargs) -> str:
  return f'data/{get_file_name(**kwargs)}.json'

from pathlib import Path

def load_data(**kwargs):
  with Path(get_data_file_name(**kwargs)).open('r') as f:
    return pd.read_json(f, orient='records')

def store_data(df: pd.DataFrame, **kwargs):
  with Path(get_data_file_name(**kwargs)).open('w+') as f:
    df.to_json(f, orient='records')

def main():
  df = (load_data if USE_EXISTING_DATA else collect_data)()
  store_data(df)
  stack_plot(df)

def main_graph():
  G_reproduction: nx.DiGraph = nx.to_directed(nx.path_graph(N))
  G_games: nx.DiGraph = nx.to_directed(nx.path_graph(N))
  G_reproduction_name = 'line'
  G_games_name = 'line'

  strategies = {}
  for node in list(G_games.nodes())[:-1]:
    strategies[node] = Strategy.AlwaysCooperate
  strategies[list(G_games.nodes())[-1]] = Strategy.AlwaysDefect

  kwargs = {
    'GAME_GRAPH': G_games_name,
    'REPRODUCTION_GRAPH': G_reproduction_name,
  }

  df = load_data(**kwargs) if USE_EXISTING_DATA else collect_data_graph(G_games, G_reproduction, strategies)
  store_data(df, **kwargs)
  stack_plot(df, **kwargs)

USE_EXISTING_DATA = False
if __name__ == '__main__':
  main()