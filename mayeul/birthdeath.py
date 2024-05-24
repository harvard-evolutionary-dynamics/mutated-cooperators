# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:08:14 2024

@author: mayeu
"""


"""
Moran process : Birth-Death process
"""

import random
import numpy as np

"""
Utility functions :
They are quite similar
"""

# recursive function to know who will give birth
# weighted_population is F*i where F is the fitnesses of every strategy, i the amount of every strategy
def who_give_birth_old(weighted_population,who):
    return random.choices(list(range(len(weighted_population))), weights=weighted_population)[0]

def who_give_birth(weighted_population,who):
    if(len(weighted_population)==1):
        return who
    
    total = sum(weighted_population)
    r = random.random()*total
    
    if(r <= weighted_population[0]):
        return who
    else:
        weighted_population = weighted_population[1:]
        return who_give_birth(weighted_population,who+1)
  
# recursive function to know who will die
# i is the amount of every strategy
def who_death_sentence_old(i,who):
    return random.choices(list(range(len(i))), weights=i)[0]

def who_death_sentence(i,who):
    if(len(i)==1):
        return who

    total = sum(i)
    r = random.random()*total
    
    if(r <= i[0]):
        return who
    else:
        i = i[1:]
        return who_death_sentence(i,who+1)

# exact same function but to choose in what will mutate the child
def which_mutation(mutation_vector, who):
    if(len(mutation_vector)==1):
        return who
    
    total = sum(mutation_vector)
    r = random.random()*total
    
    if(r<= mutation_vector[0]):
        return who
    else:
        mutation_vector = mutation_vector[1:]
        return which_mutation(mutation_vector,who+1)
  
"""
Main class : Population

We always consider N-1 individuals who follow strategy 1
and one invader who follows strategy 2.

However, you can change the initial "self.i" to generalize with n other mutants
"""

      
class Population:
    def __init__(self,population_size, payoff_matrix, mutation_matrix):
        self.N = population_size
        self.payoff_matrix = payoff_matrix # payoff matrix of all the strategies
        self.nstrat = len(payoff_matrix) # number of strategies
        self.i = [self.N-1 if w == 0 else 1 if (w == 1) else 0 for w in range(self.nstrat)] # amount of every strategy
        self.mutation_matrix = mutation_matrix # m_ij = the probability that the strategy i mutates into strategy j
        
    def give_birth(self):
        N = self.N
        i = self.i
        payoff_matrix = self.payoff_matrix
        nstrat = self.nstrat
        
        # frequencies:
        #x = np.array([i[iterate]/(N-1) for iterate in range(nstrat)])
        
        
        # F is the array of all the fitnesses
        F = np.zeros(nstrat)
        for strat in range(nstrat):
            for j in range(nstrat):
                if j==strat:
                    xj_ = (max(0, i[j]-1)/(N-1)) if i[strat]-1 > 0 else 0
                    F[strat] += xj_*payoff_matrix[strat][j]
                else:
                    xj = (i[j]/(N-1)) if i[strat] > 0 else 0
                    F[strat] += xj*payoff_matrix[strat][j]
                    
        #F = payoff_matrix@x, if we didn't care about the detail on frequency
        
        # when we play the donation game F[i] can be negative, so we take the exponential
        F[np.where(F == 0)] = -np.inf
        F = np.exp(F)
        # print(F)
        # input()

        weighted_population = F*i
        
        # who?
        who = who_give_birth(weighted_population, 0)
        return who
    
    def death_sentence(self,who_birth):
        i = (self.i).copy()
        
        # the new born replaces another individual other than its parent:

        i[who_birth] = i[who_birth] - 1
        
        # remark : it doesn't apply to the real "self.i"
        
        # who?
        who = who_death_sentence(i, 0)
        return who

    def finish_round(self,who_birth,who_death):
        #birth:
        mutation_vector = self.mutation_matrix[who_birth]
        which = who_birth if self.i[1] == 0 else which_mutation(mutation_vector,0)
        
        self.i[which] = self.i[which] + 1
        
        #death:
        self.i[who_death] = self.i[who_death] - 1

    # is there only one strategy left?
    def is_dominated(self):
        i = self.i
        nstrat = self.nstrat
        
        not_extincted = sum([int(i[w] != 0) for w in range(nstrat)])
        return not_extincted == 1

# analytical calculation from Evolutionary Dynamics page 99 equation (6.12)
def analytical_fixation_probability(k, N, payoff_matrix):
    assert len(payoff_matrix)==2
    
    if(k==0):
        return 0
    
    gammas=[]
    
    sum_k = 0
    sum_total = 0
    
    for i in range(1,N):
        fi = np.exp((i-1)/(N-1)*payoff_matrix[1,1] + (N-i)/(N-1)*payoff_matrix[1,0])
        gi = np.exp(i/(N-1)*payoff_matrix[0,1] + (N-i-1)/(N-1)*payoff_matrix[0,0])
        
        alpha_i = (i*fi/(i*fi+(N-i)*gi))*(N-i)/N
        beta_i = ((N-i)*gi/(i*fi+(N-i)*gi))*i/N
        
        gamma_i = beta_i/alpha_i
        gammas.append(gamma_i)
        
        if(i<k):
            sum_k += np.prod(gammas)
            
        sum_total += np.prod(gammas)        
    
    return (1+sum_k)/(1+sum_total)