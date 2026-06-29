#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import gaussian_kde, norm
#%matplotlib qt

np.random.seed(10)
r = 0
n = 2000
x = np.random.normal(0, 1, n)
noise_base = np.random.normal(0, 1, n) 

# Initial Y
y = (x - x.mean()) * r + noise_base * np.sqrt(1 - r**2)

# Slice mask
mask = (x > -0.095) & (x < 0.05)
y_slice = y[mask]

vec = np.arange(-5, 5, 0.01)
kde = gaussian_kde(y_slice)

def update(val):
    r = slider.val
    
    # Recalculate Y using the fixed noise_base
    y = (x - x.mean()) * r + noise_base * np.sqrt(max(0, 1 - r**2))
    
    # Update slice
    y_slice = y[mask]
    
    # Update dots
    dots.set_ydata(y)
    dots_m.set_ydata(y_slice)
    
    mu, std = norm.fit(y_slice)
    kde_new = norm.pdf(vec, mu, std)
    
    # Update KDE logic
    if abs(r) == 1:
        dist.set_xdata(vec * 0)
    else:
        kde = gaussian_kde(y_slice)
        #kde_new = kde(vec)
        dist.set_xdata(kde_new / max(kde_new) * 2)
        
    fig.canvas.draw_idle()

fig = plt.figure(figsize=(5, 5))
ax_m = fig.add_axes([0.2, 0.3, 0.7, 0.6])
ax_s1 = fig.add_axes([0.2, 0.125, 0.7, 0.05])

slider = Slider(ax_s1, 'correlation', 
                valmin=-1, valmax=1, valstep=0.01, valinit=0,
                color='#2c8cde') 

dots, = ax_m.plot(x, y, '.', markersize=10, color='black', alpha=0.05, markeredgecolor = 'None')
dots_m, = ax_m.plot(x[mask], y[mask], '.', markersize=10, 
                    color='#756bb1', alpha=1)

# Static Normal Distribution (Reference)
mu, std = norm.fit(y_slice)
pdf_vals = norm.pdf(vec, mu, std)
dist_com, = ax_m.plot(pdf_vals / max(pdf_vals) * 2, vec, color='k', linestyle='--')

# Dynamic KDE
kde_vals = kde(vec)
# dist, = ax_m.plot(kde_vals / max(kde_vals) * 2, vec, color='#756bb1')
pdf_vals = norm.pdf(vec, mu, std)
dist, = ax_m.plot(pdf_vals / max(pdf_vals) * 2, vec, color='#756bb1')

ax_m.set_xlim(-5, 5); ax_m.set_ylim(-5, 5)



slider.on_changed(update)
plt.show()


# %% [markdown]
"""
## Hello 
### test
$$a+b$$
$x^2 - x + 2^3$

```python
import numpy as np
```

check

"""

# %%
