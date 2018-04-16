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
