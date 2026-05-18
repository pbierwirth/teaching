#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:55:32 2026

@author: philipp
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm

n = 1000
d = 0.5

co = np.arange(-4,8,0.01)

fpr = norm.sf(co, loc=0, scale=1)
tpr = norm.sf(co, loc=d, scale=1)

cu_s = np.array([fpr[500], tpr[500]])

fig = plt.figure(figsize=(5,8))
ax_roc = fig.add_axes([0.1,0.25,0.8,0.5])
ax_norm = fig.add_axes([0.1,0.75,0.8,0.2])

ax_sli = fig.add_axes([0.2,0.05,0.7,0.05])
ax_sli_d = fig.add_axes([0.2,0.11,0.7,0.05])

roc, = ax_roc.plot(fpr, tpr)
ax_roc.plot(tpr, tpr, color='black', linestyle='--', alpha=0.5)
pnt = ax_roc.scatter(
    cu_s[0], cu_s[1],100
)

ax_norm.plot(co, norm.pdf(co, loc = 0, scale=1))
pdf, = ax_norm.plot(co, norm.pdf(co, loc = d, scale=1))
ax_norm.set_axis_off()
co_line = ax_norm.axvline(0, color='black')


slider = Slider(ax_sli, label='cut off', valmin=-5, valmax=5, valinit=0, valstep=0.1)
slider_d = Slider(ax_sli_d, label='effect size', valmin=-4, valmax=8, valinit=d, valstep=0.1)


def update(val):
    val = slider.val
    val_d = slider_d.val

    # update
    x = norm.sf(val, loc=0, scale=1)
    y = norm.sf(val, loc=val_d, scale=1)
    # draw
    pnt.set_offsets([x,y])   # call the method
    co_line.set_data(([val,val],[0,1]))

    # update 2
    pdf.set_ydata(norm.pdf(co, loc = val_d, scale=1))

    # update roc
    roc.set_xdata(norm.sf(co, loc=0, scale=1))
    roc.set_ydata(norm.sf(co, loc=val_d, scale=1))


    fig.canvas.draw_idle()

slider.on_changed(update)
slider_d.on_changed(update)
plt.show()

np.trapz(tpr,fpr,0.01)