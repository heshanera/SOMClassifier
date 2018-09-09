import matplotlib.pylab as plt
# import sompy as sompy
import pandas as pd
import numpy as np
from time import time
import sompy

### A toy example: two dimensional data, four clusters

dlen = 200



Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
# test = [[1,2],[4,5]]
# Data1 = pd.DataFrame(test)

# test[] = [1,0.73,Very Good,E,VS1,61.5,57,3492,5.78,5.83,3.57
# 2,0.76,Ideal,F,VS2,61.6,55,2725,5.88,5.9,3.63
# 3,1.01,Good,G,SI2,63.8,53,4185,6.41,6.31,4.06
# 4,2.02,Very Good,F,SI1,62.7,59,17530,7.97,8.03,5.02
# 5,1.23,Ideal,J,SI1,63.1,58,4959,6.8,6.74,4.27
# 6,0.73,Very Good,D,SI2,60.4,59,2703,5.8,5.86,3.52
# 7,0.4,Ideal,G,SI1,60.9,55,873,4.82,4.79,2.93
# 8,2,Ideal,H,SI1,63.9,54,18440,7.92,8,5.1
# 9,1.51,Premium,J,SI1,61.2,62,6976,7.36,7.32,4.49]


Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] + .42*np.random.rand(dlen,1))[:,0]


Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
Data2.values[:,1] = (-1*Data2.values[:,0][:,np.newaxis] + .62*np.random.rand(dlen,1))[:,0]

Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
Data3.values[:,1] = (.5*Data3.values[:,0][:,np.newaxis] + 1*np.random.rand(dlen,1))[:,0]


Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + .5*np.random.rand(dlen,1))[:,0]
#
#
Data1 = np.concatenate((Data1,Data2,Data3,Data4))

print(len(Data1))


#
fig = plt.figure()
plt.plot(Data1[:,0],Data1[:,1],'ob',alpha=0.2, markersize=4)
fig.set_size_inches(7,7)
#
#
mapsize = [20,20]
# this will use the default parameters, but i can change the initialization and neighborhood methods
som = sompy.SOMFactory.build(Data1, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
 # verbose='debug' will print more, and verbose=None wont print anything
som.train(n_job=1, verbose='info')


plt.show()
