#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


# In[2]:


class MonteCarloAgent(BaseAgent):
        
    def update(self, states_list, actions_list, rewards_recev_list):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        state_count = len(states_list) - 1
        Gt_next = 0
        for t in range(state_count, -1, -1):
            Gt = rewards_recev_list[t] + self.gamma * Gt_next
            self.Q_sa[states_list[t], actions_list[t]] += self.learning_rate * (Gt - self.Q_sa[states_list[t], actions_list[t]])
            

        # TO DO: Add own code



# In[3]:


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    for steps in range(n_timesteps):
        s = env.reset()
        states_list = []
        actions_list = []
        rewards_recev_list = []
        for t in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            actions_list.append(a)
            rewards_recev_list.append(r)
            states_list.append(s_next)
            # print(len(states_list),' state')
            # print(len(rewards_recev_list),' reward')

            if done:
                #print('reward received is ', r, ' episode value is ', t,' and the step value is ',steps)
                
                #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
                break

        #print('going for update at ',steps)    
        pi.update(states_list, actions_list, rewards_recev_list)

        if steps % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            print('Mean return for timestep mc ',steps,' is ',eval_return)
            eval_timesteps.append(steps)
            eval_returns.append(eval_return)

    # TO DO: Write your Monte Carlo RL algorithm here!
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 


# In[4]:


def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns,eval_timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print(eval_returns)


# In[5]:


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




