import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm
#%matplotlib qt

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Define the domain
vec = np.arange(0, 12, 0.01)

# Initial parameters
init_prior_mu = 8
init_prior_scale = 1
init_lik_mu = 4
init_lik_scale = 1

# Calculate initial distributions
prior = norm.pdf(x=vec, loc=init_prior_mu, scale=init_prior_scale)
likelihood = norm.pdf(x=vec, loc=init_lik_mu, scale=init_lik_scale)
posterior = prior * likelihood
posterior /= np.sum(posterior) * 0.01  # Normalize to make it a true PDF (dx = 0.01)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.4) # Make room for sliders

# Initial plots (normalized for visual comparison)
l_prior, = ax.plot(vec, prior, label='Prior', color='blue')
l_lik, = ax.plot(vec, likelihood, label='Likelihood', color='orange')
l_post, = ax.plot(vec, posterior, label='Posterior', linestyle='--', color='green')

ax.set_ylim(0, 1.0)
ax.legend()
ax.set_title('Bayesian Updating')

# Define slider axes [left, bottom, width, height]
ax_prior_mu = plt.axes([0.15, 0.25, 0.65, 0.03])
ax_prior_scale = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_lik_mu = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_lik_scale = plt.axes([0.15, 0.05, 0.65, 0.03])

# Create sliders
s_prior_mu = Slider(ax_prior_mu, 'Prior $\mu$', 0.0, 12.0, valinit=init_prior_mu)
s_prior_scale = Slider(ax_prior_scale, 'Prior $\sigma$', 0.1, 5.0, valinit=init_prior_scale)
s_lik_mu = Slider(ax_lik_mu, 'Likelihood $\mu$', 0.0, 12.0, valinit=init_lik_mu)
s_lik_scale = Slider(ax_lik_scale, 'Likelihood $\sigma$', 0.1, 5.0, valinit=init_lik_scale)

# Update function to be called when sliders change
def update(val):
    # Get current slider values
    p_mu = s_prior_mu.val
    p_scale = s_prior_scale.val
    l_mu = s_lik_mu.val
    l_scale = s_lik_scale.val
    
    # Recalculate distributions
    new_prior = norm.pdf(x=vec, loc=p_mu, scale=p_scale)
    new_lik = norm.pdf(x=vec, loc=l_mu, scale=l_scale)
    new_post = new_prior * new_lik
    
    # Normalize posterior
    area = np.sum(new_post) * 0.01
    if area > 0:
        new_post /= area
        
    # Update line data
    l_prior.set_ydata(new_prior)
    l_lik.set_ydata(new_lik)
    l_post.set_ydata(new_post)
    
    # Dynamically adjust y-axis limit based on max peak
    ax.set_ylim(0, max(np.max(new_prior), np.max(new_lik), np.max(new_post)) * 1.1)
    
    fig.canvas.draw_idle()

# Register the update function with each slider
s_prior_mu.on_changed(update)
s_prior_scale.on_changed(update)
s_lik_mu.on_changed(update)
s_lik_scale.on_changed(update)

plt.show()
