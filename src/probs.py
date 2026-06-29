#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:53:33 2026

@author: philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


empty_space = np.zeros((6,6))

fig   = plt.figure(figsize=(4,4))
# space = fig.add_axes((0.1,0.1,0.7,0.7))
space = fig.add_subplot()
space.pcolor(empty_space, cmap='grey_r',edgecolors='black')
space.set_xticks([]); space.set_yticks([])

#%%
cmap = ListedColormap(np.array([[1,1,1],[213/255,62/255,79/255]]))

sample_space = np.zeros((6,6))
sample_space[3,3] = 1

fig   = plt.figure(figsize=(4,4))
space = fig.add_subplot()
space.pcolor(sample_space, cmap=cmap,edgecolors='black')
space.set_xticks([]); space.set_yticks([])

#%%
cmap = ListedColormap(np.array([[1,1,1],[213/255,62/255,79/255]]))

sample_space = np.zeros((6,6))
sample_space[2:4, 2:4] = 1

fig   = plt.figure(figsize=(4,4))
space = fig.add_subplot()
space.pcolor(sample_space, cmap=cmap,edgecolors='black')
space.set_xticks([]); space.set_yticks([])

#%%
cmap = ListedColormap(np.array([[1,1,1],[213/255,62/255,79/255]]))

sample_space = np.zeros((6,1))
sample_space[2:4, 2:4] = 1

fig   = plt.figure(figsize=(4,4))
space = fig.add_subplot()
space.pcolor(sample_space, cmap=cmap,edgecolors='black')
space.set_xticks([]); space.set_yticks([])

#%% joint probability
cmap1 = ListedColormap(np.array([[1,1,1],[213/255,62/255,79/255]]))
cmap2 = ListedColormap(np.array([[1,1,1],[50/255,136/255,189/255]]))

event_a = np.zeros((6,6))
event_a[3:,:3] = 1

event_b = np.zeros((6,6))
#event_b[:,:] = np.nan
event_b[2:5,1:4] = 1

fig   = plt.figure(figsize=(4,4))
space = fig.add_subplot()
space.pcolor(event_a, cmap=cmap1,edgecolors='black')
space.pcolor(event_b, cmap=cmap2,edgecolors='black', alpha=0.6)
space.set_xticks([]); space.set_yticks([])
