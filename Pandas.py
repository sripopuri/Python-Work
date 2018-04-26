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





























