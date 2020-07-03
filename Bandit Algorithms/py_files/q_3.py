exec(open("core.py").read())

import matplotlib.pyplot as plt
import numpy as np


def q3_1(mu=0.5,delta=0.05,num_sims =100):
  
  bandits = [mu,mu+delta]
  arms = [BernoulliArm(p) for p in bandits]
  k = len(arms)
  
  horizons = np.arange (1000,5200,200)
  algo_names = ["UCB1","EXP3"]

  ucb_regret =[]
  exp3_regret =[]
  for i in range(len(horizons)):
    n = horizons[i]
    learning_rate = np.sqrt(2*np.log(k) / (n*k))
    exp3 = Exp3(learning_rate, [])
    ucb = UCB1([], [])
    ucb_result = test_algorithm(ucb, arms, num_sims, n)
    exp3_result = test_algorithm(exp3, arms, num_sims, n)
    ucb_regret.append((np.array([(t+1)*max(bandits) for t in range(n)]) - np.array(ucb_result[0]))[-1])
    exp3_regret.append((np.array([(t+1)*max(bandits) for t in range(n)]) - np.array(exp3_result[0]))[-1])
  
  regrets = [ucb_regret,exp3_regret]

  for i in range(len(regrets)):
    plt.plot(horizons,regrets[i],label = algo_names[i])
  
  plt.legend(loc="best")
  plt.suptitle("UCB & EXP3: Pseudo Regret vs Horizon")
  plt.title("(Horizon step: 200)",fontsize = 8)
  plt.xlabel('Horizon', fontsize=8)
  plt.ylabel('Pseudo Regret', fontsize=8)
  plt.show()  

def q3_2(mu=0.5,delta=0.05,num_sims =100,horizon=10000):

  bandits = [mu,mu+delta]
  arms = [BernoulliArm(p) for p in bandits]
  n= horizon

  learning_rates =np.arange (0,0.11, 0.01)
  algo_names = ["UCB1","EXP3"]

  ucb_regret =[]
  exp3_regret =[]
  for i in range(len(learning_rates)):
    exp3 = Exp3(learning_rates[i], [])
    ucb = UCB1([], [])
    ucb_result = test_algorithm(ucb, arms, num_sims, n)
    exp3_result = test_algorithm(exp3, arms, num_sims, n)
    ucb_regret.append((np.array([(t+1)*max(bandits) for t in range(n)]) - np.array(ucb_result[0]))[-1])
    exp3_regret.append((np.array([(t+1)*max(bandits) for t in range(n)]) - np.array(exp3_result[0]))[-1])
  
  regrets = [ucb_regret,exp3_regret]

  for i in range(len(regrets)):
    plt.plot(learning_rates,regrets[i],label = algo_names[i])
  
  plt.legend(loc="best")
  plt.suptitle("UCB1 & EXP3: Pseudo Regret vs Learning Rate")
  plt.title("(Horizon: 10000, lr step: 0.01)", fontsize=8)
  plt.xlabel('Learning Rate', fontsize=8)
  plt.ylabel('Pseudo Regret', fontsize=8)
  plt.show()  

def q3_3(mu =0.5,horizon =10000,num_sims=100):

  deltas =np.arange (0.05,0.55, 0.05)
  n = horizon
  #learning_rate = np.sqrt(2*np.log(k) / (n*k))
  learning_rate = 0.01
  algo_names = ["UCB1","EXP3"]
  ucb_regret =[]
  exp3_regret =[]

  for i in range(len(deltas)):
    bandits = [mu,mu+deltas[i]]
    arms = [BernoulliArm(p) for p in bandits]
    
    exp3 = Exp3(learning_rate, [])
    ucb = UCB1([], [])
    ucb_result = test_algorithm(ucb, arms, num_sims, n)
    exp3_result = test_algorithm(exp3, arms, num_sims, n)
    ucb_regret.append((np.array([(t+1)*max(bandits) for t in range(n)]) - np.array(ucb_result[0]))[-1])
    exp3_regret.append((np.array([(t+1)*max(bandits) for t in range(n)]) - np.array(exp3_result[0]))[-1])
  
  regrets = [ucb_regret,exp3_regret]

  for i in range(len(regrets)):
    plt.plot(deltas,regrets[i],label = algo_names[i])
  
  plt.legend(loc="best")
  plt.suptitle("UCB & EXP3: Pseudo Regret vs Delta")
  plt.title("(Horizon: 10000, Delta step: 0.05)", fontsize=8)
  plt.xlabel('Delta', fontsize=8)
  plt.ylabel('Pseudo Regret', fontsize=8)
  plt.show()  