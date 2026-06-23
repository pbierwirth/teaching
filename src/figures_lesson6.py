import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde, ttest_1samp


plt.style.use('slide_stlye.mplstyle')

x = np.arange(-60,60,0.01)
choice_a = norm.pdf(x, loc = 0,scale=10)
choice_b = norm.pdf(x, loc = 8,scale=10)
choice_c = norm.pdf(x, loc = 16,scale=10)

fig1 = plt.figure(figsize=(3,2))
ax = fig1.add_subplot()
ax.plot(x, choice_a)
ax.plot(x, choice_b)
ax.plot(x, choice_c)
ax.set_ylim(0,0.08)
ax.set_yticks([])
ax.set_xticks([])
plt.savefig(fname = 'aleatoric_distros.png', dpi=300)

#%%

def kalman_filter(x):

    v = np.zeros(len(x))
    v[0] = 0

    for i in range(1,len(x)):
        alpha = 1/i**0.6
        v[i] =  v[i-1] +  alpha * (x[i-1] - v[i-1])
    
    return v

pool_a = np.random.normal(0,10,1000)
pool_b = np.random.normal(8,10,1000)
pool_c = np.random.normal(16,10,1000)

xa = np.random.choice(pool_a, 400)
xb = np.random.choice(pool_b, 400)
xc = np.random.choice(pool_c, 400)

fig2 = plt.figure(figsize=(3,2))
ax = fig2.add_subplot()
plt.plot(kalman_filter(xa), linewidth=1)
plt.plot(kalman_filter(xb), linewidth=1)
plt.plot(kalman_filter(xc), linewidth=1)
ax.set_ylabel('Erwarteter Gewinn')
ax.set_xlabel('Durchgang')
ax.set_yticks([])
ax.set_xticks([])
plt.savefig(fname = 'Kalman.png', dpi=300)

#%%

n = 50
group = np.random.normal(0.3,1,n)
sem = group.std()/np.sqrt(n)


x   = np.arange(group.mean()-5*sem, group.mean()+5*sem,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=sem)

fig3 = plt.figure(figsize=(3,4))
ax = fig3.add_subplot()
ax.scatter(1, group.mean(), 80, edgecolor='black')
ax.errorbar(1, group.mean(), yerr = sem*1.96, linestyle='--', color='C0')
ax.axhline(0, color='white')

ax.plot((1-pdf*0.1),x, color='C0', linewidth=0.5)
ax.fill_between((1-pdf*0.1),x, color='C0', alpha=0.1)

ax.set_xticks([])
ax.set_xlim(0,2)
ax.set_ylim(-0.5,1)
fig3.savefig(fname = 'ttest_0_epi.png', dpi=300)


std = group.std()
x   = np.arange(group.mean()-5*std, group.mean()+5*std,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=std)


fig4 = plt.figure(figsize=(3,4))
ax   = fig4.add_subplot()
ax.errorbar(1, group.mean(), yerr = std*1.96, linestyle='--', color='C1', zorder=1)
ax.scatter(1, group.mean(), 80, edgecolor='black',zorder=1, color='C1')

ax.plot((1-pdf*0.8),x, color='C1', linewidth=0.5,zorder=1)
ax.fill_between((1-pdf*0.8),x, color='C1', alpha=0.1, zorder=1)

ax.axhline(0, color='white', zorder=1)

ax.scatter(1.3+0.2*(np.random.random(len(group))-0.5),group, 4, color='C1')
ax.set_xlim(0,2)
ax.set_xticks([])
fig4.savefig(fname = 'ttest_0_allo.png', dpi=300)



std = group.std()
x   = np.arange(group.mean()-5*std, group.mean()+5*std,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=std)

fig5 = plt.figure(figsize=(3,4))
ax   = fig5.add_subplot()
ax.errorbar(2, group.mean(), yerr = std*1.96, linestyle='--', color='C1', zorder=1)
ax.scatter(2, group.mean(), 40, edgecolor='black',zorder=1, color='C1')

ax.plot((2+pdf*0.8),x, color='C1', linewidth=0.5,zorder=1)
ax.fill_between((2+pdf*0.8),x, color='C1', alpha=0.1, zorder=1)

ax.axhline(0, color='white', zorder=1)

ax.scatter(1.6+0.2*(np.random.random(len(group))-0.5),group, 4, color='C1')

ax.scatter(1, group.mean(), 40, edgecolor='black')
ax.errorbar(1, group.mean(), yerr = sem*1.96, linestyle='--', color='C0')
ax.axhline(0, color='white')

x   = np.arange(group.mean()-5*sem, group.mean()+5*sem,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=sem)
ax.plot((1-pdf*0.1),x, color='C0', linewidth=0.5)
ax.fill_between((1-pdf*0.1),x, color='C0', alpha=0.1)


ax.set_xlim(0,3)
ax.set_xticks([0])
plt.show()
fig5.savefig(fname = 'ttest_0_allo_epi.png', dpi=300)


#%%


n = 800
group = np.random.normal(0.3,1,n)
sem = group.std()/np.sqrt(n)


std = group.std()
x   = np.arange(group.mean()-5*std, group.mean()+5*std,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=std)

fig5 = plt.figure(figsize=(3,4))
ax   = fig5.add_subplot()
ax.errorbar(2, group.mean(), yerr = std*1.96, linestyle='--', color='C1', zorder=1)
ax.scatter(2, group.mean(), 40, edgecolor='black',zorder=1, color='C1')

ax.plot((2+pdf*0.8),x, color='C1', linewidth=0.5,zorder=1)
ax.fill_between((2+pdf*0.8),x, color='C1', alpha=0.1, zorder=1)

ax.axhline(0, color='white', zorder=1)

ax.scatter(1.6+0.2*(np.random.random(len(group))-0.5),group, 4, color='C1')

ax.scatter(1, group.mean(), 40, edgecolor='black')
ax.errorbar(1, group.mean(), yerr = sem*1.96, linestyle='--', color='C0')
ax.axhline(0, color='white')

x   = np.arange(group.mean()-5*sem, group.mean()+5*sem,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=sem)
ax.plot((1-pdf*0.02),x, color='C0', linewidth=0.5)
ax.fill_between((1-pdf*0.02),x, color='C0', alpha=0.1)


ax.set_xlim(0,3)
ax.set_xticks([0])
plt.show()
fig5.savefig(fname = 'ttest_0_Huge_N.png', dpi=300)




n = 12
group = np.random.normal(0.3,1,n)
sem = group.std()/np.sqrt(n)


std = group.std()
x   = np.arange(group.mean()-5*std, group.mean()+5*std,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=std)

fig5 = plt.figure(figsize=(3,4))
ax   = fig5.add_subplot()
ax.errorbar(2, group.mean(), yerr = std*1.96, linestyle='--', color='C1', zorder=1)
ax.scatter(2, group.mean(), 40, edgecolor='black',zorder=1, color='C1')

ax.plot((2+pdf*0.8),x, color='C1', linewidth=0.5,zorder=1)
ax.fill_between((2+pdf*0.8),x, color='C1', alpha=0.1, zorder=1)

ax.axhline(0, color='white', zorder=1)

ax.scatter(1.6+0.2*(np.random.random(len(group))-0.5),group, 4, color='C1')

ax.scatter(1, group.mean(), 40, edgecolor='black')
ax.errorbar(1, group.mean(), yerr = sem*1.96, linestyle='--', color='C0')
ax.axhline(0, color='white')

x   = np.arange(group.mean()-5*sem, group.mean()+5*sem,0.01)
pdf = norm.pdf(x, loc = group.mean(), scale=sem)
ax.plot((1-pdf*0.2),x, color='C0', linewidth=0.5)
ax.fill_between((1-pdf*0.2),x, color='C0', alpha=0.1)


ax.set_xlim(0,3)
ax.set_xticks([0])
plt.show()
fig5.savefig(fname = 'ttest_0_samll_N.png', dpi=300)