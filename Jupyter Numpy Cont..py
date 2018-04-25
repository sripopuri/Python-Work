
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


x =  np.array([1,2,3,4,5])


# In[6]:


x


# In[7]:


x > 6 


# In[8]:


x<3


# In[9]:


x != 3


# In[10]:


2*x


# In[11]:


x*2


# In[12]:


x**2


# In[13]:


2*x == x**2


# In[14]:


np.random.RandomState(0)


# In[15]:


rng = np.random.RandomState(0)


# In[16]:


rng


# In[17]:


x = rng.randint(10,size=(3,4))


# In[18]:


x


# In[19]:


x < 6 


# In[20]:


np.sum(x < 6)


# In[21]:


np.sum(x >= 6)


# In[22]:


np.sum(x > 6)


# In[23]:


np.sum(x == 6)


# In[24]:


print(x)


# In[25]:


np.count_nonzero(x < 6)


# In[26]:


np.sum(x<6)


# In[27]:


np.sum(x<6, axis = 1)


# In[28]:


x<6


# In[30]:


np.sum(x<6, axis = 0)


# In[31]:


np.any(x < 6,axis = 0)


# In[32]:


np.any(x < 6,axis = 1)


# In[33]:


np.all(x < 6,axis = 0)


# In[34]:


np.all(x < 6,axis = 1)


# In[36]:


np.any((x < 6)&(x>1))


# In[37]:


np.any((x < 6)&(x>1),axis = 0)


# In[38]:


x


# In[39]:


np.all((x < 6)&(x>1),axis = 0)


# In[40]:


np.all((x < 6)&(x>1),axis = 1)


# In[41]:


np.any((x < 6)&(x>1),axis = 1)


# In[42]:


np.sum((x<6)&(x>1))


# In[43]:


(x<6)&(x>1)


# In[44]:


x


# In[45]:


x[x<5]


# In[46]:


x[x<6]


# In[48]:


x= np.random.randint(100,size=10)


# In[49]:


x


# In[52]:


[x[3],x[4]]


# In[53]:


ind = [0,5,8,6]


# In[54]:


x[ind]


# In[57]:


ind = np.array([[3, 7],[4, 5]])


# In[58]:


x[ind]


# In[61]:


x = np.arange(12).reshape((3,4))


# In[62]:


x


# In[63]:


row = np.array([0,1,2])


# In[69]:


col = np.array([3,2,0])


# In[70]:


x[row,col]


# In[71]:


x


# In[72]:


x[2,[2,0,1]]


# In[73]:


x[1:,[2,0,1]]


# In[74]:


x = np.array([2,1,4,3,5])


# In[75]:


np.sort(x)


# In[76]:


np.argsort(x)


# In[77]:


x


# In[78]:


x[np.argsort(x)]

