import matplotlib.pylab as plt
import numpy as np
import sompy
import csv
from sklearn.preprocessing import normalize


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
            line_count += 1

    print(f'Processed {line_count} lines.')

diamondData = normalize(diamondData, axis=0, norm='max')
mapsize = [60,60]
som = sompy.SOMFactory.build(
    diamondData, mapsize, mask=None, mapshape='planar',
    lattice='rect', normalization='var', initialization='pca',
    neighborhood='gaussian', training='batch', name='sompy'
)
som.train(n_job=2, verbose='info')

# Generating Heatmaps for all variables
som.component_names = ['Carat','Cut','Color','Clarity','Depth','Table','Price','x','y','z']
v = sompy.mapview.View2DPacked(60, 60, 'Features',text_size=8)
v.show(som, what='codebook', cmap='jet', which_dim='all', col_sz=6)

# Generating the hitmap
h = sompy.hitmap.HitMapView(50, 50, 'hitmap', text_size=1, show_text=True)
h.show(som)

# Generating the U-Matrix representation
u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
UMAT = u.show(som, distance2=20, row_normalized=False, show_data=True, contooor=True, blob=False)

# Generating Heatmaps for individual variable with the scale
attributes = ['Carat','Cut','Color','Clarity','Depth','Table','Price','x','y','z']
codebook = som.codebook.matrix
msz0, msz1 = som.codebook.mapsize
attr = 0; # attribute index from the 'attributes' array
mp = codebook[:, attr].reshape(msz0, msz1)
fig, (ax0) = plt.subplots(nrows=1)
im0 = ax0.imshow(mp[::-1], norm=None, cmap="jet", vmin=0, vmax=3.5)
ax0.set_title(attributes[attr])
# colorbar scale
fig.colorbar(im0, ax=ax0)
plt.tight_layout()
plt.show()
