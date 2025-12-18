#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


# In[2]:


class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):

        Q_succ_max = np.max(self.Q_sa[s_next])
        G_target = r + self.gamma * Q_succ_max
        self.Q_sa[s, a] += self.learning_rate * (G_target - self.Q_sa[s, a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    
    # TO DO: Write your Q-learning algorithm here!
    done = False
    s = env.reset()
    for steps in range(n_timesteps):
        a = agent.select_action(s, policy, epsilon, temp)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next, done)
        s = s_next
        #env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        if done:
            print('Reached goal state in ',steps,' steps')
            #env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
            s = env.reset()
            
        steps_inc = steps + 1
        if steps_inc % eval_interval == 0:
            eval_return = agent.evaluate(eval_env) 
            eval_timesteps.append(steps_inc)
            eval_returns.append(eval_return)


        
    if plot:
       print('in plot') 
       #env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps) 


# In[3]:


def test():
    
    n_timesteps = 10000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.03
    temp = 0.9
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)
if __name__ == '__main__':
    test()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




