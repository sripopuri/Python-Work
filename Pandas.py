# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:45:32 2018

@author: sriha
"""
 
import pandas as pd
pd.__version__
import numpy as np
np.__version__

data = pd.Series([1,2,3,4,5])
data
type(data)
data.size
data.shape
 
data.values.shape
data.index
type(data.index)
type(data.values)
type(data)

data[1]
data.index[3]
data[1:3]

pd.Series([1,2,4,6],dtype = float)

data = pd.Series([1,2,3,5],index = ['a','b','c','d'],dtype = float)
data
data['a']

type(data.index)

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}

population = pd.Series(population_dict)
population

population['Texas']

population['Illinois':'Texas']
data
data = pd.Series([1,2,3,4,5,6])
data[2:4]


pd.Series(16,index = [1,2,3,4,5])

pd.Series({2:'a', 1:'b', 3:'c'}, index=[1,3])
pd.Series({2:'a', 1:'b', 3:'c'}, index=[1,4])

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}

area = pd.Series(area_dict) 
area
population

states= pd.DataFrame({'Population':population,'Area':area})
states
states.index
states.values
type(states.values)
type(states.index)
states.columns

states['Population']

pd.DataFrame({'Population':population})

data = [{'a':i,'b':2*i,'c':3*i}for i in range(4)]
pd.Series(data)

pd.DataFrame([{'a':1,'b':2},{'b':3,'c':4}])


np.random.rand(3,2)
pd.DataFrame(np.random.rand(3,2),columns = ['a','b'],index = [0,1,2])

np.zeros(3)

ind = pd.Index([1,2,3,4,5,6])
ind
ind[5]
ind[::2]
ind[::1]

print(ind.size,ind.shape,ind.ndim,ind.dtype)

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA

indA & indB
indA | indB
indA ^ indB

import pandas as pd
data = pd.Series([0.25,0.5,0.75,1.0],index =['a','b','c','d'])
data
data['b']
'c' in data
'z' in data
data.keys()
data.values
data.index
list(data.items())
data['e'] = 01.25
data
data['a':'c']
data.values[1:3]

data1 = pd.Series([1,2,3,4],index = [1,2,3,4])
data1[1:3]
data1

data
data[(data > 0.1) & (data < 0.6)]

data = pd.Series(['a','b','c'],index=[1,3,5])
data
data[1]
data[0:2]
data.loc[1]
data.loc[1:3]
data.iloc[1:3]
data

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area,'pop':pop})
data
data['area']
data.pop
data['pop']
data['density'] = data['pop']/data['area']
data

data.values
data.T

data.values[0]
data

data.iloc[0:3,0:2]
