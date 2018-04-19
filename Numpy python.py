print("Hello Python!")

print(7)

import numpy as np
np.__version__

x = 4
x = "Four"

L = list(range(10))
L

type(L)
type(L[0])

L2 = [str(c) for c in L]
type(L2[4])

L3 =[1,"Harish",True]

a = [type(i) for i in L3] 
a
a[0]

import array as ar
L = list(range(10))
A = ar.array('i',L)
A

B = ar.array('q',L)
B
B[1]
type(B[1])

import numpy as np
n = np.array([0,1,2,3,4])

c = np.array([0,1,2,3,3],dtype = 'float32')
c

l = np.array([range(i,i+3) for i in [2,4,6]],dtype = 'float32')
l

i = np.array([list(range(10))])
i

z = np.zeros(10,dtype = int)
z

np.ones((3,5),dtype = float)
np.full((4,4),3.14)

arr = np.array([range(0,20,2)])
arr
np.arange(0,20,3)

np.arange(0,1,((1-0)/5))
np.linspace(0,1,5)
np.linspace(0,20,3)
np.linspace(0,20,4)
np.linspace(0,20,6)


np.random.random?

np.random.random((3,4))
np.random.normal(0,1,(5,5))

np.random.randint(0,10,(3,3))

np.eye(3)
np.ones((3,3),dtype = int)

e = np.empty(3)

np.zeros(10,dtype = np.int16)

# Numpy array attributes
x1 = np.random.randint(10,size = 4)
x1

np.random.randint?

x2 = np.random.randint(10,size = (2,3))
x2

x3 = np.random.randint(10,size = (2,3,2))
x3
x3 = np.random.randint(10, size=(3, 4, 5))
x3

np.ndim(x3)
np.size(x3)
np.shape(x3)

type(x3)
x3.dtype

x3.itemsize
x3.nbytes

x3
x3[0,0]
x3[2,3,2]
x3[2,-3,2] = 6*2
x3


np.array([0,1,2.2],dtype= np.int16)

x1
x1[1:3]

x1 = np.random.randint(10,size = 10)
x1[1:7:2]
x1

x1[1:7:2]

y = np.arange(10)
y
y[:5]
y[5:]

y[4:7]


y
y[2:4]
y[-1]
y[-3:-1]

y[::2]
y[::3]
y[::10]

y
y = np.random.randint(20,size = (3,5))
y

y[1,]
y[:,:3]
y
y[:,1]

y[::2]
y[3::2]

y[::1]
y[::-1]

y[5::-2]
y

y
y[:]
x2
x3

x2[,] # Error
x2[:,:]
x2
x2[:2,:2]
x2[0,]
x2
x2[::-1,::-1]


x2 = np.random.randint(20,size = (3,5))
x2
print(x2)
x2[0:2,:2]
x2[1:3,0:2]
x2[:,:]
x2[1:3,2:4]
x2
subx2 = x2[:2,:2]
subx2
subx2[0,0] = 16
subx2
x2

subx2 = x2[:2,:2].copy()
subx2
subx2[0,0] = 3
subx2
x2


import numpy as np

grid = np.arange(1,10)
grid
type(grid)
grid = grid.reshape((3,3))
grid
type(grid)

x = np.array([1,2,3])
x
x.reshape((3,1))

x
x[:,np.newaxis]

x = np.array([1,2,3])
y = np.array([3,2,1])
np.concatenate((x,y))
np.concatenate?

x
y
np.concatenate((x[:,np.newaxis],y[:,np.newaxis]))
np.vstack((x,y))
np.shape(np.vstack((x,y)))
np.hstack((x,y))
np.shape(np.hstack((x,y)))

np.vstack((grid,x))
grid
np.concatenate((grid,grid))

np.shape(grid)
np.shape(np.concatenate((x,y)))
np.ndim(np.concatenate((x,y)))

np.shape(np.concatenate((grid,grid)))
np.concatenate((grid,grid))
np.concatenate((grid,grid),axis = 1)

x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
np.split?
x
y1, y2 = np.split(x,[2])
y1
y2

grid = np.arange(16).reshape((4,4))
grid
upper, lower = np.vsplit(grid,[2])
upper
lower

left, right = np.hsplit(grid,[2])
left
right

np.random.seed(1601)
np.random.randint(6,size = 10)
s = range(10)
s

 def compute_reciprocals(values):
   output = np.empty(len(values))
   for i in range(len(values)):
     output[i] = 1.0 / values[i]
  return output

np.random.seed(0)    
values = np.random.randint(1, 10, size=5)
values
compute_reciprocals(values)

np.empty(5)
np.empty(10)

big_array = np.random.randint(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)

np.arange(5)/np.arange(1,6)
np.arange(5)*np.arange(1,6)
np.arange(5)+np.arange(1,6)

x = np.arange(9).reshape((3,3))
x
2**x # 2 power x
x**x # x power x

x
x+2
x-2
x*4
x/5
x//2
x%2
x/2
-x

(0.5*x + 1)**2

np.add(x,2)

np.arange(-2,4)
np.abs(np.arange(-2,4))

z = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])    
z
np.abs(z)

np.pi
theta = np.linspace(0,np.pi,5)
theta

np.sin(theta)
np.cos(theta)
np.tan(theta)

np.arccos(-1)

x = np.array([1,2,3])
x
np.exp(x)
np.exp2(x)
np.power(3,x)
np.power(x,3)

np.log(x)
np.log2(x)
np.log10(x)
np.log1p(1) # more precise values when x is small


from scipy import special as sp
sp.gamma(5)
sp.gamma(2)
sp.gamma(6)

import numpy as np

x = np.arange(5)
y = np.empty(5)
np.multiply(x,10,out=y)

y = np.zeros(10)
y[::2]
np.power(2,x,out = y[::2])
y
