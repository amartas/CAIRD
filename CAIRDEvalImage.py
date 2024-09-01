#!/usr/bin/env python
# Author: Aidan Martas


import CAIRDIngest
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300

inputdir = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/SN/"
n_imgs = 100
outputdir = "/home/entropian/Documents/CAIRD/CAIRDDatasets/SortedData/SNSorted/"

CAIRDIngest.Reviewer(inputdir, n_imgs, outputdir)




"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


height, width = 21, 21

# Create meshgrid
X, Y = np.meshgrid(np.arange(width), np.arange(height))

# Create surface plot
def HeightPlot(channel_data, channel_name, ax):
    ax.plot_surface(X, Y, channel_data, cmap="gray")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Brightness")
    ax.set_title(f"3D Surface Plot of {channel_name} Channel")

# Create subplots
fig = plt.figure(figsize=(18, 6))

# Plot channels
ax1 = fig.add_subplot(131, projection="3d")
HeightPlot(red_channel, "Science", ax1)

ax2 = fig.add_subplot(132, projection="3d")
HeightPlot(green_channel, "Reference", ax2)

ax3 = fig.add_subplot(133, projection="3d")
HeightPlot(blue_channel, "Difference", ax3)

# Show the plots
plt.tight_layout()
plt.show()
"""