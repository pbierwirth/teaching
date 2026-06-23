import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('slide_stlye.mplstyle')


# 1. Set seed for reproducibility
np.random.seed(42)
n_samples = 20

# 2. Define the highly correlated covariance matrix
# High variance along the diagonal (50), but an extremely tight correlation (49.8)
cov = [[50, 49.8], 
       [49.8, 50]]

# 3. Define shifted means (slightly perpendicular to the correlation line)
mean_group_A = [-1, 1]
mean_group_B = [1, -1]

# Create groups
group_A = np.random.multivariate_normal(mean_group_A, cov, n_samples)
group_B = np.random.multivariate_normal(mean_group_B, cov, n_samples)

x = np.arange(-20,20, 0.01)

dens_A1 = norm.pdf(x = x, loc = group_A[:,0].mean(), scale = group_A[:,0].std())
dens_B1 = norm.pdf(x = x, loc = group_B[:,0].mean(), scale = group_B[:,0].std())

dens_A2 = norm.pdf(x = x, loc = group_A[:,1].mean(), scale = group_A[:,1].std())
dens_B2 = norm.pdf(x = x, loc = group_B[:,1].mean(), scale = group_B[:,1].std())



fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.scatter(group_A[:,0], np.ones(len(group_A)), 50, edgecolors = 'black')
ax.plot(x, dens_A1*8+1.2)
ax.scatter(group_B[:,0], np.ones(len(group_A))-0.5,50, edgecolors = 'black')
ax.plot(x, dens_B1*8+1.2)
ax.set_xlabel('Merkmal A')
ax.set_yticks([])
ax.set_ylim(0,2)
ax.set_title('Gruppenvergleich Merkmal A')
fig.savefig('./images/merkmal_a.png', dpi=600)



fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.scatter(group_A[:,1], np.ones(len(group_A)), 50, edgecolors = 'black')
ax.plot(x, dens_A2*8+1.2)
ax.scatter(group_B[:,1], np.ones(len(group_A))-0.5, edgecolors = 'black')
ax.plot(x, dens_B2*8+1.2)
ax.set_xlabel('Merkmal B')
ax.set_yticks([])
ax.set_ylim(0,2)
ax.set_title('Gruppenvergleich Merkmal B')
fig.savefig('./images/merkmal_b.png', dpi=600)


d1 = (group_A[:,0].mean() - group_B[:,0].mean())/np.std((group_A[:,0],group_B[:,0]))
d2 = (group_A[:,1].mean() - group_B[:,1].mean())/np.std((group_A[:,1],group_B[:,1]))

#%%


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.scatter(group_A[:,0], group_A[:,1],50, edgecolors='black')
ax.plot(x,20+200*dens_A1, color = 'C0')
ax.plot(x,20+200*dens_B1, color = 'C1')
ax.scatter(group_B[:,0], group_B[:,1],50,edgecolors='black')
ax.plot(20+200*dens_A2, x, color = 'C0')
ax.plot(20+200*dens_B2, x, color = 'C1')

ax.set_xlabel('Merkmal A')
ax.set_ylabel('Merkmal B')
ax.set_title('Multivariate Betrachtung')

fig.savefig('./images/multivariate_plot.png', dpi = 600)

#%% 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

x = np.concat((group_A, group_B))
y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
lr.fit(x,y);

y_hat = lr.intercept_ + lr.coef_[0][0] * x[:,0] + lr.coef_[0][1] * x[:,1] 

x_clf = np.arange(-6,6,0.01)
dens_A_clf = norm.pdf(x_clf, loc=y_hat[y==0].mean(), scale = y_hat[y==0].std()) 
dens_B_clf = norm.pdf(x_clf, loc=y_hat[y==1].mean(), scale = y_hat[y==1].std()) 

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.scatter(y_hat[y==0], np.ones(n_samples),100, edgecolor = 'black')
ax.plot(x_clf, 1.02+ 0.05*dens_A_clf, color='C0')
ax.scatter(y_hat[y==1], np.ones(n_samples),100, edgecolor = 'black')
ax.plot(x_clf, 1.02+ 0.05*dens_B_clf, color='C1')
ax.set_ylim(0.98,1.1)
ax.set_yticks([])
ax.set_xlabel('Classifier-Wert')
fig.savefig('./images/classifier.png', dpi=600)

d = (y_hat[y==0].mean() - y_hat[y==1].mean())/ ((np.std(y_hat[y==1])+np.std(y_hat[y==0]))/2)