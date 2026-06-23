import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import numpy as np

plt.style.use('slide_stlye.mplstyle')

#%%
N1 = int(1e+7)
N2 = int(1e+7)
es = 1
x = np.arange(130,210,0.01)

pdf_a = norm.pdf(x, loc = 170, scale=10)

fig1 = plt.figure()
plt.plot(x, pdf_a)
plt.fill_between(x, pdf_a, alpha=0.1)
plt.xlabel('Körpergröße [cm]')
plt.ylabel('Häufigkeit')
plt.ylim(0, 0.08)
plt.savefig(fname='normal_dist_single.png', format='png', dpi=600)

np.random.seed(8)
population = np.random.normal(170, 10, N1)
sample     = np.random.choice(population, 20)
kde = gaussian_kde(sample)

fig1 = plt.figure()
plt.plot(x, kde(x))
plt.fill_between(x, kde(x), alpha=0.1)
plt.xlabel('Körpergröße [cm]')
plt.ylabel('Häufigkeit')
plt.ylim(0, 0.08)
plt.savefig(fname='normal_dist_sample20.png', format='png', dpi=600)

sample.mean()
# draw sample

#%%
n = 2000
N1 = int(1e+7)
N2 = int(1e+7)
es = 1
x = np.arange(130,210,0.01)
pdf_a = norm.pdf(x, loc = 170, scale=10)

fig1 = plt.figure()
ax = fig1.add_subplot()
ax.plot(x, pdf_a)
ax.fill_between(x, pdf_a, alpha=0.1)
ax.set_ylim(0, 0.08)
ax.set_xticks([]); ax.set_yticks([])
ax.axvline(170, ymin=0, ymax = pdf_a[np.argmin(np.abs(x-170))]/0.08)
ax.annotate(text = '$\mu = 170$',xy=(170,0.045), fontsize=25)
plt.savefig(fname='normal_dist_stripped.png', format='png', dpi=200)

population = np.random.normal(170, 10, N1)
for i in range(5):
    sample     = np.random.choice(population, n)
    kde = gaussian_kde(sample)

    fig1 = plt.figure()
    ax = fig1.add_subplot()
    plt.plot(x, kde(x))
    plt.fill_between(x, kde(x), alpha=0.1)
    ax.set_ylim(0, 0.08)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(sample.mean(), ymin=0, ymax = kde(x)[np.argmin(np.abs(x-sample.mean()))]/0.08)
    ax.annotate(text = rf'$\bar{{x}} = {sample.mean():.2f}$',xy=(sample.mean(),0.01+kde(x)[np.argmin(np.abs(x-sample.mean()))]), fontsize=25)
    plt.savefig(fname=f'fig_{i}_n={n}.png', format='png', dpi=200)



#%%
means = []
for i in range(int(1e+5)):
    sample = np.random.choice(population, n)
    means.append(sample.mean())
means = np.array(means)

x = np.arange(160,180,0.01)
kde = gaussian_kde(means)

fig1 = plt.figure()
plt.plot(x, kde(x))
plt.fill_between(x, kde(x), alpha=0.1)
plt.xlabel(r'Stichprobenmittelwerte $\bar{x}$')
plt.ylabel('Häufigkeit')
#plt.ylim(0, 0.5)
plt.savefig(fname=f'SEM_dist_n={n}.png', format='png', dpi=400)




#%% one feature in two groups
N1 = int(1e+7)
N2 = int(1e+7)
es = 1
x = np.arange(-4,4,0.01)

pdf_a = norm.pdf(x, loc = 0, scale=1)
pdf_b = norm.pdf(x, loc = es,scale=1)


fig1 = plt.figure()
plt.plot(x, pdf_a, label='group_a')
plt.plot(x, pdf_b, label='group_b')
plt.legend()
plt.xlabel('feature value')
plt.ylabel('frequency')
plt.ylim(0,1)

#%%

ssizes = np.arange(10,1000,10)
means = np.zeros((int(1e+3),len(ssizes)))
stds = np.zeros((int(1e+3),len(ssizes)))

for i in range(int(1e+3)):
    for j in range(len(ssizes)):
        sample = np.random.choice(population, ssizes[j])
        means[i,j] = sample.mean()
        stds[i,j]  = sample.std(ddof=1)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(ssizes, 10/np.sqrt(ssizes))
ax.tick_params(axis='y', colors='#66c2a5')
ax.set_ylabel('Standardfehler (SEM)', color = '#66c2a5')
ax.set_xlabel('Stichprobengröße (n)')
plt.twinx()
plt.scatter(ssizes, stds.mean(axis=0),10, alpha=0.2, color='#fc8d62')
plt.axhline(10, color='#fc8d62')
plt.tick_params(axis='y', colors='#fc8d62')
plt.ylim(0,11)
plt.ylabel('Standardabweichung',color='#fc8d62')
plt.savefig(fname = 'sem_std.png', dpi=200)



#%% Generate simulation of small studies
group_a = np.random.normal(0,1,N1)
group_b = np.random.normal(es,1,N1)

fig = plt.figure()

for i in range(100):
    sample_size = 10
    idx = np.random.choice(np.arange(0,N1),(sample_size,2))
    diff = group_a[idx[:,0]].mean() - group_b[idx[:,1]].mean()
    plt.subplot(3,1,1)
    plt.scatter(i,diff, color='#66c2a5')
    plt.ylim(-1.8,1.8)

    sample_size = 30
    idx = np.random.choice(np.arange(0,N1),(sample_size,2))
    diff = group_a[idx[:,0]].mean() - group_b[idx[:,1]].mean()
    plt.subplot(3,1,2)
    plt.scatter(i,diff, color='#66c2a5')
    plt.ylim(-1.8,1.8)

    sample_size = 200
    idx = np.random.choice(np.arange(0,N1),(sample_size,2))
    diff = group_a[idx[:,0]].mean() - group_b[idx[:,1]].mean()
    plt.subplot(3,1,3)
    plt.scatter(i,diff, color='#66c2a5')
    plt.ylim(-1.8,1.8)



kde_a,kde_b = gaussian_kde(group_a[idx[:,0]]), gaussian_kde(group_b[idx[:,0]])

plt.plot(x, kde_a(x), label='group_a', linewidth=0.25, color = '#66c2a5')
plt.plot(x, kde_b(x), label='group_b', linewidth=0.25, color = '#fc8d62')
plt.legend()
plt.xlabel('feature value')
plt.ylabel('frequency')
plt.ylim(0,1)
