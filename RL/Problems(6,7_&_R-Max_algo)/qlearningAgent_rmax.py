# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgentRMax(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()
        
        ## Only keeps the state which are known(i.e visited >=m no of times for each legal action)
        self.known_state = []
        
        ## Keeps the number of times a (state,action) pair has occured
        ## key:(state,action) , value: No of times
        self.visited_times = util.Counter()
        
        ## This keeps which action has become known for a state
        ## key: state, value: actions that are vistied >=m no of times
        ## helped to reduce complexity
        self.known_state_action={}
        
        ## keeps the reward for a (state,action) pair
        ## gets updated with time
        self.rewards_mk = util.Counter()
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        
        if len(legalActions)==0:
          return 0
          
        dic = util.Counter()
        
        for action in legalActions:
          dic[action] = self.getQValue(state, action)
        
        return dic[dic.argMax()] 
    
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        best_action = None
        max_val = -999999
        
        for action in legal_actions:
          
          q_value = self.q_values[(state, action)]
          
          if max_val < q_value:
            max_val = q_value

            poss_actions = [k[1] for k,v in self.q_values.items() if (k[0],v) == (state,max_val)]
            best_action = random.choice(poss_actions)
       
        return best_action
    
    
      
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        
        for i in legal_actions:
            if (state,i) not in self.visited_times.keys():
                self.visited_times[(state,i)] = 0
        
        ## For known state we follow the policy
        ## We can explore with epsilon prob.
        ## We are also exploring in the unknown states
        ## We have the choice
        
        
        if state in self.known_state:
            return self.getPolicy(state)
            
            ## For exploring
            '''
            explore = util.flipCoin(self.epsilon)
        
            if explore:
                return random.choice(legal_actions)
            else:
                return self.getPolicy(state)
            '''
                
        
        ## We are in unknown state
        else:
             
             ## if a state is not visited for any action then we choose a random action as r(s,a) is same for all a
             state_list = [] 
             
             for i in self.visited_times.keys():
                 state_list.append(i[0])
             state_list= set(state_list)
             
             ## For state which is not seen yet for any actions
             if state not in state_list:
                return random.choice(legal_actions)
             
             ## Here we take that action which is visited least no of times
             ## There may be more than one action which has occured least no of times
             ## So again we choose a action randomly among those
             ## When we have only one action with least occurance, random.choice has only one action to choose   
             
             min_visit= 99999999
             for i in self.visited_times.keys():
                   if i[0]==state:
                      if self.visited_times[i]< min_visit:
                         min_visit= self.visited_times[i]
                         poss_actions = [k[1] for k,v in self.visited_times.items() if (k[0],v) == (state,min_visit)]
             
             return random.choice(poss_actions)
                                

    
    def update(self, state, action, nextState, reward):
        
        ## Used the update_q function written below
        
        ## In unknown state
        if state not in self.known_state:
           
           self.visited_times[(state,action)]+=1
           self.rewards_mk[(state,action)]+=reward
           
           self.update_q(state, action, nextState, self.rewards_mk[(state,action)]) 
         
        ## In known state   
        else:
           
           updated_reward = self.rewards_mk[(state,action)]/self.visited_times[(state,action)]
           self.update_q(state, action, nextState, updated_reward)   
        
        ## Now to append a state to known states if it becomes known
        
        legal_actions = self.getLegalActions(state)
        
        if self.visited_times[(state,action)]>=self.m:
         
           try:
              self.known_state_action[state].append(action)
           except:
              self.known_state_action[state] = [action]
            
        ## Need to satisfy three conditions for my case
       
        if state not in self.known_state:
           if state in self.known_state_action.keys() and len(self.known_state_action[state]) == len(legal_actions):
              self.known_state.append(state)      
        
    
    ## Following function is called in update function
    
    def update_q(self, state, action, nextState, reward):
        
        ## In terminal state
        if not nextState:
           self.q_values[(state, action)] = reward
       
       ## For other states
        else:   
           nextState_val = self.discount * self.getValue(nextState)  
           self.q_values[(state, action)] = reward + nextState_val
        
           
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

