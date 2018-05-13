import numpy as np

L = np.random.random(100)
sum(L)
np.std(L) 
np.sum(L)

big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array) 

np.min(big_array)
np.max(big_array)

M = np.random.random((3,4))
M
M.sum()
M.sum(axis = 0)
M.max(axis = 1)
np.prod(M)
np.mean(M,axis = 0)
np.argmin(M)
M
np.min(M)
np.max(M)
np.argmax(M)

x = np.arange(1,13).reshape((3,4))
x

np.argmax(x)
np.argmin(x)
np.median(x)

import pandas as pd
data = pd.read_csv('data/president_heights.csv')

a = np.arange(3)
b = np.arange(3)[:,np.newaxis]
b
a + b

M = np.ones((2,3))
M
a = np.arange(3)
a
M + a

M = np.arange(3).reshape((3,1))    
M
a
M + a

M = np.ones((3,2))
M
a = np.arange(3).reshape((3,1))
a
M + a

import numpy as np
X = np.random.random((10,3))
X
X.mean(0)

X_centered = X - X.mean(0)
X_centered
X_centered.mean(0)

x = np.linspace(0,5,50)
y = np.linspace(0,5,50)[:,np.newaxis]

z = np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)
%matplotlib inline
import matplotlib.pyplot as plt    

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],cmap='viridis')
plt.colorbar();

import pandas as pd
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values

np.arange(365)
