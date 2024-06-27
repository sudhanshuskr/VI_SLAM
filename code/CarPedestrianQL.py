#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


## Q9 ##
states = [] ## number of states will now be 14x8x8x2 - 8 options for each pedestrian, 2 for flag - 0 if no double step, 1 if a double step
rewards = {}
for i in range(7):
    for j in range(2):
        carState = str(i) + str(j)
        for k in range(8):
            ped1_State = str(k)
            for l in range(8):
                ped2_State = str(l)
                for m in range(2):
                    state = carState + str(k) + str(l) + str(m)
                    states.append(state)
                    if(carState == '00' or carState == '01'):
                        rewards[state] = 0
                    elif((carState == '30' and ped1_State=='4') or (carState == '31' and ped1_State=='3')):
                        rewards[state] = -100
                    elif((carState == '40' and (ped2_State=='3' or (m==1 and ped1_State=='4'))) or (carState == '41' and (ped2_State=='4'  or (m==1 and ped1_State=='3')))):
                        rewards[state] = -100
                    elif((carState == '50' and m==1 and ped2_State=='3') or (carState == '51' and m==1 and ped2_State=='4')):
                        rewards[state] = -100
                    elif(carState == '60' or carState == '61'):
                        rewards[state] = 10
                    else:
                        rewards[state] = -1

actions = [1,2,3] ## 1 for 1 step down, 2 for 2 steps doen, 3 for diagonal movement


# In[3]:


def getIndex(state):
    return int(state[4]) + 2*int(state[3]) + 16*int(state[2]) + 128*int(state[1]) + 256*int(state[0])


# In[4]:


# Q-table formation

Qtable = np.zeros((len(states)+2*8*8*2, len(actions))) # initialized by 0


# In[5]:


256*7 + 128*1 + 16*7 + 2*7 + 1*1


# In[6]:


Qtable.shape


# In[7]:


def simulate(state, action):
    carState = state[:2]
    if(carState == '60' or carState == '61'):
        return '70'+state[2:]
    c1 = int(state[0])
    c2 = int(state[1])
    c3 = int(state[2])
    c4 = int(state[3])
    if(action == 1):
        return (str(c1+1) + str(c2) + str(min(c3+1, 7)) + str(min(c4+1, 7)) + '0')
    elif(action == 2):
        return (str(min(c1+2, 6)) + str(c2) + str(min(c3+1, 7)) + str(min(c4+1, 7)) + '1')
    else:
        return (str(c1+1) + str(np.abs(c2-1)) + str(min(c3+1, 7)) + str(min(c4+1, 7)) + '0')


# In[8]:


def pick_action(state, epsilon):
    a = random.uniform(0, 1)
    if(a < epsilon):
        ##exploration, take random action
        act = random.choice(actions)
    else:
        st_idx = int(state[4]) + 2*int(state[3]) + 16*int(state[2]) + 128*int(state[1]) + 256*int(state[0])
        act = np.argmax(Qtable[st_idx]) + 1
    return act


# In[9]:


def QLearning(gamma=0.9, alpha=0.25, epsilon=0.2):
    num_episodes = 10000
    qt0 = []
    qt1 = []
    qt2 = []
    for i in range(num_episodes):
        # print(i)
        carState = random.choice(['00','01'])
        ped1_State = random.choice(['0','1','2','3','4','5','6','7'])
        ped2_State = random.choice(['0','1','2','3','4','5','6','7'])
        # mState = random.choice(['0','1'])
        mState = '0'
        state = carState + ped1_State + ped2_State + mState
        while(state[:2] != '60' and state[:2] != '61'):
            # print("s ", state)
            action = pick_action(state, epsilon)
            # print("current state : ", state)
            # print("act : ", action)
            s_next = simulate(state, action)
            state_idx = getIndex(state)
            # state_idx = int(state[4]) + 2*int(state[3]) + 16*int(state[2]) + 128*int(state[1]) + 256*int(state[0])
            # print("next state : ", s_next)
            # print("state_idx : ", state_idx)
            Qtable[state_idx][action-1] += alpha * (rewards[s_next] + gamma*np.max(Qtable[getIndex(s_next)]) - Qtable[state_idx][action-1])
            state = s_next
        qt0.append(Qtable[getIndex('30220')][0])
        qt1.append(Qtable[getIndex('30220')][1])
        qt2.append(Qtable[getIndex('30220')][2])
    return qt0, qt1, qt2


# In[10]:


qts = QLearning()


# In[11]:


x = np.arange(10000)+1
plt.plot(x, qts[0])
plt.show()


# In[13]:


x = np.arange(10000)+1
plt.plot(x, qts[1])
plt.show()


# In[14]:


x = np.arange(10000)+1
plt.plot(x, qts[2])
plt.show()


# In[12]:


## D1 - 30220 and 30221
print(Qtable[getIndex('30220')], Qtable[getIndex('30221')])


# In[21]:


Qtable[getIndex('50771')], Qtable[getIndex('00771')] ## car in last row


# In[14]:


st = '50541'
st[:2]


# In[15]:


'70'+st[2:]


# In[ ]:




