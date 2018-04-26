# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:50:52 2018

@author: sriharish
"""

# Import
import numpy as np
np.__version__
np.show_config()


x = np.array(10)
x
x[0]

x = np.zeros(10)
x[0]
x

x.nbytes
x.shape
x.size
x.itemsize

print("Memory occupied by x array is %d bytes" % (x.nbytes))

?np.add

n = np.zeros(10)
n
n[4] = 1
n

vector = np.arange(10,50)
vector

print(n)
vector.size
vector.shape
vector[39]
vector[0:40]
vector[-1]
vector[1::-2]
vector

m = np.arange(0,9).reshape((3,3))
m

v = np.array([1,2,0,0,4,0])
v
v[v!=0]
np.argwhere(v)

nz = np.nonzero(v)
nz

I = np.eye(3)
I

r = np.random.randint(1,4,size=(3,3,3))
r
np.random.random((3,3,3))

r = np.random.random((10,10))
r
np.min(r)
np.max(r)

v = np.random.random(size=30)
v
np.mean(v)

z = np.zeros((3,3))
z
z[0,] = 1
z[2,] = 1
z[:,0] = 1 
z[:,2] = 1
z

z = np.ones((10,10))
z
z[1:-1,1:-1] = 0
z

z
z = np.pad(z,pad_width = 1,mode = 'constant',constant_values = 0)
z

0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1

m = np.diag(1+ np.arange(4),k=-1)
m

c = np.ones((8,8))
c
c[0::1,1::2] = 0
c

z = np.random.random((6,7,8))
z
z.shape
z.size
z.nbytes

np.unravel_index(100,(6,7,8))
z[np.unravel_index(100,(6,7,8))]


np.tile(np.array([[0,1],[1,0]]),(3,3))

r = np.random.random((5,5))
r
r = (r - np.min(r))/(np.max(r)-np.min(r))
r

a = np.random.random((5,3))
b = np.random.random((3,2))
a*b
?np.dot
np.dot(a,b)
a@b
