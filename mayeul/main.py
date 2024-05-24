# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:39:52 2024

@author: mayeu
"""

import birthdeath as bd
import numpy as np
import matplotlib.pyplot as plt

# Game parameters:
N = 10
b = 5
c = 1

# Simulation parameters:
nb_games = 5000

# Statistics:
mus = np.linspace(0,1,20)
print(mus)
fixation_probabilities = []

# do we play neutral games ?
neutral = False

for mu in mus:
    
        if(neutral):
    # neutral:
            payoff_matrix = np.array([[1,1,1],[1,1,1],[1,1,1]])
            mutation_matrix = np.eye(len(payoff_matrix))
    
    # with some mu:
        else:
            payoff_matrix = np.array([[b-c,-c,b-c],[b,0,0],[b-c,0,(b-c)/2]])
            # ALLC, ALLD, TFT
            mutation_matrix = np.array([[1-mu, 0,mu],[0,1,0],[0,0,1]])
        
        # number of dominations counted per strategy
        nb_dominations = np.zeros(len(payoff_matrix))
        for iterate_game in range(nb_games):
            
            
            
            # new population :
            population = bd.Population(N,payoff_matrix,mutation_matrix)
            
            # we play until a strategy dominates all the others
            while(not population.is_dominated()):
                who_birth = population.give_birth()
                who_death = population.death_sentence(who_birth)
                population.finish_round(who_birth, who_death)

            # we want to find who dominates to do statistics
            strat = 0
            found = False
            while(strat<population.nstrat and not found):
                if(population.i[strat] == population.N):
                    nb_dominations[strat] = nb_dominations[strat] + 1
                    found = True
                strat = strat + 1
        
        #after all of the games have been played, we calculate the fixation probability
        fixation_probability = nb_dominations / nb_games
        fixation_probabilities.append(fixation_probability)

        print("Fixation probability is", 100*fixation_probability, "%" )


# plots:
fixation_probabilities = np.matrix(fixation_probabilities)
y1 = [fixation_probabilities[i,0] for i in range(len(fixation_probabilities))]
y2 = [fixation_probabilities[i,1] for i in range(len(fixation_probabilities))]
y3 = [fixation_probabilities[i,2] for i in range(len(fixation_probabilities))]

fig, ax = plt.subplots()
plt.stackplot(mus,y2,y3,y1,colors=['b','orange','g'],labels=['ALLD', 'TFT', 'ALLC'])

plt.title(f"Birth-Death process for population size N={N}, b={b}, c={c}", )
plt.legend()
ax.set_xlabel("Mu")
ax.set_ylabel("Domination %")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()