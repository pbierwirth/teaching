#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:21:25 2026

@author: philipp
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(35)

n = 8
x = np.random.normal(0,1,n)


#%% define generative model

y_true = 2.5*x + 4
noise  = np.random.normal(0,5,n) 
y      = y_true + noise 



#%%
params = []
error  = [] 
for i in range(1,7):
    params.append(np.polyfit(x,y,i))


x_vec = np.arange(-2,2,0.01)
fig = plt.figure(figsize=(8,4))
ax = fig.subplots(2,3)
for j,axis in enumerate(ax.flatten()) :

    axis.scatter(x,y, 
                100,
                edgecolor = 'black', 
                color     = '#2c8cbe')
    fit = np.polyval(params[j],x_vec)
    axis.plot(x_vec, fit, color='black')
    axis.set_xlim(-2,2); axis.set_ylim(-10,20)
    
    axis.spines[ ['top','bottom','right','left']].set_visible(False)
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_title(f'{j+1}order polynomial')
    
    error.append(np.sum((np.polyval(params[j],x)-y)**2))
    
plt.tight_layout()

error = np.array(error)

plt.figure(figsize=(4,4))
plt.plot(np.arange(1,7,1), error, color='black', zorder=0)
plt.scatter(np.arange(1,7,1), error, 
            100,
            color='#2c8cbe',
            edgecolor='black', zorder=1)
plt.xlabel('polynomial order')
plt.ylabel('sum of the squared error')

#%%
np.random.seed(1)
error_test  = [] 
x_test = np.random.normal(0,1,n)

y_true_test = 2.5*x_test + 4
noise  = np.random.normal(0,5,n) 
y_test      = y_true_test + noise 

x_vec = np.arange(-2,2,0.01)
fig = plt.figure(figsize=(8,4))
ax = fig.subplots(2,3)
for j,axis in enumerate(ax.flatten()) :

    axis.scatter(x_test,y_test, 
                100,
                edgecolor = 'black', 
                color     = '#de2d26')
    fit = np.polyval(params[j],x_vec)
    axis.plot(x_vec, fit, color='black')
    axis.set_xlim(-2,2); axis.set_ylim(-10,20)
    
    axis.spines[ ['top','bottom','right','left']].set_visible(False)
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_title(f'{j+1}order polynomial')
    
    error_test.append(np.sum((np.polyval(params[j],x_test)-y_test)**2))
    
plt.tight_layout()

plt.figure(figsize=(4,4))
plt.plot(np.arange(1,7,1), error, color='black', zorder=0)
plt.scatter(np.arange(1,7,1), error, 
            100,
            color='#2c8cbe',
            edgecolor='black', zorder=1)
plt.xlabel('polynomial order')
plt.ylabel('sum of the squared error (SSE)')
plt.twinx()
plt.plot(np.arange(1,7,1), np.log10(error_test), color='black',linestyle='--' ,zorder=0)
plt.scatter(np.arange(1,7,1), np.log10(error_test), 
            100,
            color='#de2d26',
            edgecolor='black', zorder=1)
plt.yticks([np.log10(100),np.log10(1000),np.log10(10000),
            np.log10(100000)],
           labels=['10e+1', '10e+2', '10e+3', '10e+4'])
plt.ylabel(r'$log_{10}(SSE)$')
plt.tight_layout()