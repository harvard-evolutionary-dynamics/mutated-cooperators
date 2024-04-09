import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import random
import pandas as pd

from enum import Enum
from typing import *

class Strategy(Enum):
  AlwaysCooperate = 'ALLC'
  TitForTat = 'TFT'
  AlwaysDefect = 'ALLD'

INDIVIDUALS = [Strategy.AlwaysCooperate, Strategy.TitForTat, Strategy.AlwaysDefect]

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

def imitation(N: int, nALLC: int, nTFT: int, M: Payoffs, mu: float):
  assert 0 <= mu <= 1, mu
  nALLD = N-nALLC-nTFT
  assert nALLC + nTFT + nALLD == N
  assert all(nX >= 0 for nX in (nALLC, nTFT, nALLD)), (nALLC, nTFT, nALLD)
  pick_individual = lambda: random.choices(population=INDIVIDUALS, weights=np.array([nALLC, nTFT, nALLD])/N)[0]
  strategy_picked_for_update = pick_individual()
  strategy_role_model = pick_individual()

  payoff_picked_for_update = M[strategy_picked_for_update, strategy_picked_for_update]
  payoff_role_model = M[strategy_role_model, strategy_picked_for_update]

  F = lambda x: np.exp(x)
  fitnesses = np.array(list(map(F, (payoff_picked_for_update, payoff_role_model))))
  total_fitness = sum(fitnesses)
  winning_strategy = random.choices(population=[strategy_picked_for_update, strategy_role_model], weights=fitnesses/total_fitness)[0]

  # Conditional mutation.
  new_strategy = winning_strategy
  if winning_strategy == Strategy.AlwaysCooperate and nALLD > 0:
    new_strategy = random.choices(population=[Strategy.TitForTat, new_strategy], weights=[mu, 1-mu])[0]
    
  return (
    nALLC + int(new_strategy == Strategy.AlwaysCooperate) - int(strategy_picked_for_update == Strategy.AlwaysCooperate),
    nTFT + int(new_strategy == Strategy.TitForTat) - int(strategy_picked_for_update == Strategy.TitForTat),
  )

def main():
  INTERVALS = 100
  TRIALS = 100
  data = []
  for N in (50,):
    for mu in np.linspace(0, 0.5, INTERVALS, endpoint=True):
      fixated_ALLD = []
      fixated_ALLC_given_ALLD_extinct = []
      for _ in range(TRIALS):
        nALLC = N-1
        nTFT = 0
        nALLD = 1
        while all(nX < N for nX in (nALLC, nTFT, nALLD)):
          nALLCp, nTFTp = imitation(N, nALLC, nTFT, M, mu=mu)
          nALLC, nTFT, nALLD = (nALLCp, nTFTp, N-(nALLCp + nTFTp))

        fixated_ALLD.append(nALLD == N)
        if nALLD == 0:
          fixated_ALLC_given_ALLD_extinct.append(nALLC == N)


      fp_ALLD = np.mean(fixated_ALLD)
      fp_ALLC_given_ALLD_extinct = np.mean(fixated_ALLC_given_ALLD_extinct)
      data.append((N, mu, fp_ALLD, fp_ALLC_given_ALLD_extinct))

  print(data)
  df = pd.DataFrame(data, columns=['N', 'mu', 'fp_ALLD', 'fp_ALLC_given_ALLD_extinct'])
  fig, ax = plt.subplots(1,2)
  sns.lineplot(df, ax=ax[0], x='mu', y='fp_ALLD', hue='N', linestyle='--', marker='o', legend=False)
  sns.lineplot(df, ax=ax[1], x='mu', y='fp_ALLC_given_ALLD_extinct', hue='N', linestyle='--', marker='o', legend=False)
  fig.savefig(f'figs/M-ft_ALLD-saptarshi.png', dpi=300, bbox_inches="tight")
  plt.show()
  # fig.show()

if __name__ == '__main__':
  main()