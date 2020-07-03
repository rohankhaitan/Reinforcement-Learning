from collections import Counter
import numpy as np

def test_algorithm(algo, arms, num_sims, horizon):
  chosen_arms = [0.0 for i in range(num_sims * horizon)]
  rewards = [0.0 for i in range(num_sims * horizon)]
  cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
  sim_nums = [0.0 for i in range(num_sims * horizon)]
  times = [0.0 for i in range(num_sims * horizon)]

  for sim in range(num_sims):
    sim = sim + 1
    algo.initialize(len(arms))
    
    for t in range(horizon):
      t = t + 1
      index = (sim - 1) * horizon + t - 1
      sim_nums[index] = sim
      times[index] = t
      
      chosen_arm = algo.select_arm()
      chosen_arms[index] = chosen_arm
      
      reward = arms[chosen_arms[index]].draw()
      rewards[index] = reward
      
      if t == 1:
        cumulative_rewards[index] = reward
      else:
        cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
      
      algo.update(chosen_arm, reward)

  cumulative_rewards= np.array(cumulative_rewards).reshape((num_sims,horizon)).tolist()
  avg_cumulative_rewards =[sum(x)/num_sims for x in zip(*cumulative_rewards)]
  
  chosen_arms = (np.array(chosen_arms).reshape((num_sims,horizon)).T).tolist()
  counts = []
  arm_count = []
  for i in range(horizon):
    temp = Counter(chosen_arms[i])
    counts.append([temp[i]/num_sims for i in range(len(arms))])
  
  for i in range(len(arms)):
      arm_count.append([v[i] for v in counts])

  return [avg_cumulative_rewards, arm_count]

