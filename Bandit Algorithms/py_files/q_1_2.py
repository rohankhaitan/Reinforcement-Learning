exec(open("core.py").read())

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np



num_sims = 1000
horizon = 1000


algo1 = EpsilonGreedy(0.1,[], [])
algo2 = UCB1([], [])
algo3 = Exp3(0.01, [])  ## learning rate = 0.01
algo4 = Exp3_gamma(0.05, [])  ## gamma = 0.5

algos = [algo1, algo2, algo3, algo4]
algo_names = ["Epsilon-greddy","UCB1","EXP3","EXP3-gamma"]

bandits_1 = [0.1,0.1,0.1,0.1,0.6]
bandits_2 = [0.1,0.2,0.5,0.8,0.95]

results =[]
arms = [BernoulliArm(p) for p in bandits_2]
n_arms = len(arms)

for algo in algos:
  results.append(test_algorithm(algo, arms, num_sims, horizon))


def pseudo_regret(result):
  regret = np.array([t*max(bandits_2) for t in range(1,horizon+1)]) - np.array(result[0])
  return regret 

def arm_count(result):
  chosen_arms = (np.array(result[2]).reshape((1000,1000)).T).tolist()
  counts = []
  arm_count = []
  for i in range(horizon):
    temp = Counter(chosen_arms[i])
    counts.append([temp[i] for i in range(len(arms))])
  
  for i in range(len(arms)):
      arm_count.append([v[i] for v in counts])

  return arm_count

def q1():
  regret = []
  for result in results:
    regret.append(pseudo_regret(result))
  for i in range(len(algos)):
    plt.plot(regret[i],label = algo_names[i])
  
  plt.legend(loc="best")
  plt.suptitle("Pseudo Regret vs Time")
  plt.title("(Horizon: 1000)",fontsize = 8)
  plt.xlabel('Time', fontsize=8)
  plt.ylabel('Pseudo Regret', fontsize=8)
  plt.show()  



def q2(): 
  
  k=0
  nrows=2
  ncols =2
  fig, ax = plt.subplots(nrows=2, ncols=2)

  for i in range(ncols):
    for j in range(nrows):
      arm_counts = results[k][1]
      for m in range(len(arms)):
        ax[i,j].plot(arm_counts[m],label ="arm-"+str(m+1))
      ax[i,j].legend(loc="best",prop={'size': 5})
      ax[i,j].set_title("Arm Count vs Time ("+algo_names[k]+")", fontsize=6)
      ax[i,j].set_xlabel('Time', fontsize=5)
      ax[i,j].set_ylabel('Arm Count', fontsize=5)
      ax[i,j].tick_params(axis='both', which='major', labelsize=5)
      ax[i,j].tick_params(axis='both', which='minor', labelsize=5)
      k=k+1
  fig.tight_layout(pad=1.0)
  plt.show()

