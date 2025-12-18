#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


# In[2]:


class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        
        T_ep = len(rewards)
        Gt_tar = 0
        list_target = []
        G_test = 0
        #print('rewards is ',rewards)
        for t in range(T_ep):
            m = min(n, T_ep - t)
            if t + m == T_ep and done:
                for i in range(m):
                    temp = self.gamma**i * rewards[t+i]
                    
                    list_target.append(temp)
                Gt_tar = sum(list_target)
                list_target = []
                
                
            else:
                
                for i in range(m):
                    temp = self.gamma**i * rewards[t+i]
                    
                    list_target.append(temp)
                Gt_tar =  sum(list_target) + (self.gamma**m * np.max(self.Q_sa[states[t + m]]))
                
                list_target = []
                
            
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (Gt_tar - self.Q_sa[states[t], actions[t]])
        
        # TO DO: Add own code
        


# In[3]:


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    T_ep = 0
    done = False
    s = env.reset()
    eps_stat = False
    while T_ep < n_timesteps:
        states_list = [s]
        actions_list = []
        rewards_recev_list = []
        eps_stat = False
        for t in range(max_episode_length):
            s_now = states_list[-1]
            a = pi.select_action(s_now, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            

            states_list.append(s_next)
            actions_list.append(a)
            rewards_recev_list.append(r)

            if (t + T_ep + 2) % eval_interval == 0:
                # print('step ',steps)
                # print('t ',t)
                eval_return = pi.evaluate(eval_env)
                print('Mean return for timestep ',(t + T_ep + 2),' is ',eval_return)
                eval_timesteps.append(t + T_ep + 2)
                eval_returns.append(eval_return)

            if done:
                #print('break hua')
                #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
                eps_stat = True
                break

        
            
        pi.update(states_list, actions_list, rewards_recev_list, done, n)
        # print(len(states_list),' state')
        # print(len(actions_list),' action')
        # print(len(rewards_recev_list),' rew')
        s = env.reset()
        T_ep += len(actions_list)

            
            
    # TO DO: Write your n-step Q-learning algorithm here!
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 


# In[4]:


def test():
    n_timesteps = 70000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 3
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns,eval_timesteps=n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print(eval_returns)


# In[5]:


if __name__ == '__main__':
    test()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




