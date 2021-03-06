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

class QLearningAgent(ReinforcementAgent):
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
        
        legal_actions = self.getLegalActions(state)
        
        if len(legal_actions)==0:
          return 0
        
        ## If we use util.counter then we can use the argMax function 
        ## to get the action for which a state has maximum Q-value
        
        dic = util.Counter()
        
        for action in legal_actions:
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
        max_val = -9999999
        
        for action in legal_actions:
          q_value = self.getQValue(state, action) 
          
          if max_val < q_value:
            max_val = q_value
        
            ## Now we can have more than one actions for a state with the max q-value
            ## We choose a random among those actions (for a state) which have same max q-value
            ## For that we can do the following. Otherwise we can just set best_action to be action 
            
            ## Slight change in the following due to complexity,both works
            #poss_actions = [k[1] for k,v in self.q_values.items() if (k[0],v) == (state,max_val)]
            poss_actions = [action for action in legal_actions if self.getQValue(state,action) == max_val]
            best_action = random.choice(poss_actions)
            #best_action = action
        
        return best_action

        ## Another eaiser way I could have used in the earlier submission 
        # legal_actions = self.getLegalActions(state)
        # max_val = self.getValue(state)
        # poss_actions = [action for action in legal_actions if self.getQValue(state,action) == max_val]
        # best_action = random.choice(poss_actions)
        #return best_action
        
        

        
		
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
        
        ## Toss a coin and based on value of epsilon set explore as true or false
        explore = util.flipCoin(self.epsilon)
        
        ## If explore is true then a random choice among the legal actions
        ## Else we follow the policy
        
        if explore:
            return random.choice(legal_actions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ## previous q-value
        prev_q_value = self.getQValue(state, action)
        
        ## previous part and current reward
        ## alpha is learning rate. It is working as weights.
        ## If we don't use alpha then Q-values will explode.
        
        ## If no nextstate-(In Terminal state)
        if not nextState:
          self.q_values[(state, action)] = (1 - self.alpha) * prev_q_value + self.alpha * reward
        
        ##For other states
        else:
          nextState_val = self.alpha * self.discount * self.getValue(nextState)
          self.q_values[(state, action)] = (1 - self.alpha) * prev_q_value + self.alpha * reward + nextState_val

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action



class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        ## Features
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0
        
        ## Approximate Q-value
        for i in features:
            q_value += features[i] * self.weights[i]
        
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        ## Features
        features = self.featExtractor.getFeatures(state, action)
        ## Compute the difference
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        
        ## Update of weight vectors
        for i in features:
            self.weights[i] = self.weights[i] + self.alpha * difference * features[i]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("Approximate Q-Learning")
            print("No of Training episodes: {0}".format(self.numTraining))
            
            

