#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[2]:


import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


# In[3]:


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        # TO DO: Add own code
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        #print(s,'s hai')
        
        Q_sa_updated = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1))) 
        max_abs_error = np.max(np.abs(Q_sa_updated - self.Q_sa[s, a]))
        self.Q_sa[s, a] = Q_sa_updated
        return (Q_sa_updated,max_abs_error)
        
        


# In[4]:


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma,threshold)

    max_iter = float('inf') 
    
    while max_iter > threshold:
        max_iter = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                curr_q_val = QIagent.Q_sa[s, a]
                s_next, r = env.model(s, a)
                QIagent.update(s, a, s_next, r)
                
                
                max_iter = max(max_iter, np.abs(curr_q_val - QIagent.Q_sa[s, a]))
    
    
 
     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    print(f'V star (s = 3) at start state is {np.max(QIagent.Q_sa[3])} ')
    done = False
    s = env.reset()
    #print(s, 'current state')
    total_reward = 0
    timestep = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        # print(s_next,' s_next hai')
        # print(r,' r wala reward')
        p_sas,r_sas = env.model(s,a)
        # print(r_sas,' r_sas wala reward')
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next
        total_reward += r
        timestep += 1
        p_sas, r_sas = env.model(s, a)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
    mean_reward_per_timestep = total_reward / timestep
    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




