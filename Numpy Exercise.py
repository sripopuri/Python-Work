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

n = np.arange(0,12)
n
n[3:7] = -n[3:7]
n

n = np.arange(0,12)
n
n[(n>3)&(n<8)] *= -1
n

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

sum(range(6),-1)
range(6)
sum(range(6),-1)

Z
Z**Z
2 << Z
Z/1/1
1j*Z

Z = np.arange(0,11)
Z
2 << Z
Z << 2

np.array(0)//np.array(0)
np.array([np.nan]).astype(int).astype(float)
np.array([np.nan]).astype(int)

?np.random.uniform
z = np.random.uniform(-10,10,10)
z
np.ceil(z)
np.round(z)

print (np.copysign(np.ceil(np.abs(z)), z))

z1 = np.random.randint(0,10,10)
z2 = np.random.randint(0,10,10)
z1
z2
np.intersect1d(z1,z2)

np.sqrt(-1) 
np.emath.sqrt(-1)

today = np.datetime64('today', 'D')
today

Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
Z

np.ones(3)

z = np.random.uniform(0,4,10)
z
np.floor(z)
np.round(z)
np.ceil(z)-1
z-z%1 ## <- gets the decimal parts of the number
np.trunc(z)
z.astype(int)

f = np.random.randint(0,5,size = (5,5))
f
np.arange(5)

def generate():
    for x in range(10): 
        yield(x)

generate()

array = np.fromiter(generate(),dtype=float)
array

np.random.random(10)

np.linspace(0,1,11,endpoint = False)[1:]

z = np.linspace(0,1,11,endpoint = False)[1:]
z
np.sort(z)

np.arange(10)
np.add.reduce(np.arange(10))

A = np.random.randint(0,2,5)
A
B = np.random.randint(0,2,5)
B
A == B
np.array_equal(A,B)
A[3] = 1
np.array_equal(A,B)

Z = np.ones(10)
Z
Z.flags.writeable = False
Z[1] = 1
Z

r = np.random.random(10)
r
np.max(r)
np.argmax(r)
r[np.argmax(r)] = 0
r

np.linspace(0,1,5)

Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
print(Z)

np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)

Z = np.arange(100)
v = np.random.uniform(0,100)

np.min(Z-v)
index = (abs(Z-v)).argmin()
index
Z[index]
v
Z
