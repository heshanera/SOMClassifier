import matplotlib.pylab as plt
from sklearn.preprocessing import normalize
# import sompy as sompy
import pandas as pd
import numpy as np
from time import time
import sompy

import csv


cut = ['Premium','Ideal','Very Good','Good','Fair']
color = ['D','I','H','E','F','G','J']
clarity = ['VVS1','VVS2','VS1','VS2','SI1','SI2','IF','I1']

diamondData = []

with open('diamonds.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # first line
            line_count += 1
        else:
            # creating an array
            features = []
            for i in range(len(row)):
                if (i == 0):
                    continue
                if (i == 2):
                    features.append(cut.index(row[i]))
                elif (i == 3):
                    features.append(color.index(row[i]))
                elif (i == 4):
                    features.append(clarity.index(row[i]))
                else:
                    features.append(float(row[i]))

            diamondData.append(features)
            if (line_count == 35000):
                break
            line_count += 1
    print(f'Processed {line_count} lines.')

diamondData = normalize(diamondData, axis=0, norm='max')
# Data1 = pd.DataFrame(data = diamondData[0:int(line_count/2)])
# Data2 = pd.DataFrame(data = diamondData[int(line_count/2):line_count])
# diamondData = np.concatenate((Data1,Data2))

mapsize = [60,60]
som = sompy.SOMFactory.build(
    diamondData, mapsize, mask=None, mapshape='planar',
    lattice='rect', normalization='var', initialization='pca',
    neighborhood='gaussian', training='batch', name='sompy'
)
som.train(n_job=2, verbose='info')


# v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
# sompy.mapview.View2DPacked(50,50, 'Features',text_size=5).show(som)
# v.show(som, what='codebook', which_dim='all', cmap=None, col_sz=6) #which_dim='all' default
# v.save('2d_packed_test')

# from sompy.visualization.mapview import View2D
# view2D  = View2D(40,40,"rand data",text_size=7)
# view2D.show(som)

# som.component_names = ['Carat','Cut','Color','Clarity','Depth','Table','Price','x','y','z']
# v = sompy.mapview.View2DPacked(60, 60, 'Features',text_size=8)
# v.show(som, what='codebook', cmap='jet', which_dim=[6,0,7,8,9], col_sz=6)
# v.show(som, what='codebook', cmap='jet', which_dim=[6,1,2,3,4], col_sz=6)

# h = sompy.hitmap.HitMapView(50, 50, 'hitmap', text_size=1, show_text=True)
# h.show(som)

# u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
# UMAT = u.show(som, distance2=20, row_normalized=False, show_data=True, contooor=True, blob=False)

attributes = ['Carat','Cut','Color','Clarity','Depth','Table','Price','x','y','z']
codebook = som.codebook.matrix
msz0, msz1 = som.codebook.mapsize
attr = 2;
mp = codebook[:, attr].reshape(msz0, msz1)

fig, (ax0) = plt.subplots(nrows=1)
im0 = ax0.imshow(mp[::-1], norm=None, cmap="jet", vmin=0, vmax=3.5)
ax0.set_title(attributes[attr])
#colorbars
fig.colorbar(im0, ax=ax0)
plt.tight_layout()
plt.show()
