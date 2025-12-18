#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


# In[2]:


class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        Gt_tar = r + self.gamma * self.Q_sa[s_next, a_next]
        self.Q_sa[s, a] += self.learning_rate * (Gt_tar - self.Q_sa[s, a])


# In[3]:


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    # TO DO: Write your SARSA algorithm here!
    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)
    for steps in range(n_timesteps):
        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next, policy, epsilon, temp)
        pi.update(s, a, r, s_next, a_next, done)

        if done:
            print('Reached goal state in ', steps, ' steps')
            #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            s = s_next
            a = a_next

        if (steps + 1) % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_timesteps.append(steps + 1)
            eval_returns.append(eval_return)
    
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps)


# In[4]:


def test():
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)


# In[5]:


if __name__ == '__main__':
    test()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




