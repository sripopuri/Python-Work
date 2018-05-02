# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:54:23 2018

@author: sriha
"""

import numpy as np
import pandas as pd

vals1 = np.array([1, None, 3, 4])
vals1
type(vals1)
type(vals1[0])
type(vals1[1])

for dtype in ['object','int']:
    print("dtype = ",dtype)
    %timeit np.arange(1E6,dtype = dtype).sum()
    print()

vals1.min()
np.min(vals1)

vals2 = np.array([1,2,np.NaN,4])
vals1.dtype
vals2.dtype

vals2.sum(),vals2.min(),vals2.max()
np.sum(vals2)
np.nansum(vals2)
vals2

pd.Series([1,2,np.NaN,None])
series = pd.Series([1,2,3])
series.iloc[2] = np.NaN
series
series.isnull()
series.notnull()
series.dropna()
series.fillna(value = 9999)

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df

df.isnull()
df.notnull()
df.dropna(axis = 1)
df.dropna(how = 'any')
df.dropna(how = 'all')
df
df[3] = np.NaN
df
df.dropna(thresh = 3)
df
df.fillna(method = 'ffill',axis = 0)
df.fillna(method = 'ffill',axis = 1)

#df.fillna(method = 'ffill',axis = 0).fillna(method = 'ffill',axis = 1)
