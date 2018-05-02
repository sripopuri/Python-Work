# -*- coding: utf-8 -*-
"""
Created on Wed May  2 01:09:19 2018

@author: sriha
"""

import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
rng
ser = pd.Series(rng.randint(0,10,4))
ser

df = pd.DataFrame(rng.randint(0,10,(3,4)),columns = ['A','B','C','D'])
df

np.exp(ser)
df*np.pi/4
np.sin(df*np.pi/4)

area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

population/area
area.index
population.index
area.index & population.index
area.index | population.index

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A+B
A.add(B,fill_value = 1)

A = rng.randint(10, size=(3, 4))
A
df = pd.DataFrame(A, columns=list('QRST'))
df
df - df.iloc[0]

df.subtract(df['R'],axis = 0)
df
df.iloc[0, ::2]
df.iloc[::2,:]
