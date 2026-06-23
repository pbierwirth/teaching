import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

plt.style.use('slide_stlye.mplstyle')

np.random.seed(42)
x,y = make_blobs(n_samples=30, n_features = 2, centers=2, cluster_std=3)



lr = LogisticRegression()
lr.fit(x,y);
betas = np.hstack((lr.intercept_[0], lr.coef_[0]))

line  = -betas[0]/betas[2] -betas[1]/betas[2] * x[:,0] 



betas = np.hstack((lr.intercept_[0], lr.coef_[0]))
params = np.arange(betas[2]-1,betas[2]+1,0.2)
cost = []
for b in params :
    y_proba = 1/(1+np.exp(-(betas[0] + betas[1]*x[:,0] + b*x[:,1])))
    # Calculate the mean cross-entropy cost
    cost.append(-np.mean(y * np.log(y_proba) + (1 - y) * np.log(1 - y_proba)))

plt.figure(figsize=(4,4))
plt.plot(params ,cost, marker = '.', markersize=25, markeredgecolor='black', color='white')
plt.ylabel('Error')
plt.xlabel(r'$\beta_1$')
plt.savefig('./error_function.png', dpi=600)

plt.figure(figsize=(4,4))
plt.scatter(x[y==0,0],x[y==0,1],50, edgecolors='black')
plt.scatter(x[y==1,0],x[y==1,1],50,edgecolor='black')
plt.plot(x[:,0], -betas[0]/params[5]-betas[1]/params[5]*x[:,0],color='green')

plt.plot(x[:,0], -betas[0]/params[0]-betas[1]/params[0]*x[:,0],color='orange')
plt.plot(x[:,0], -betas[0]/params[-1]-betas[1]/params[-1]*x[:,0],color='red')
plt.xlim(-8,10)
plt.ylim(-5,15)

plt.xlabel('Merkmal A')
plt.ylabel('Merkmal B')
plt.savefig('./example_scatter_ml.png', dpi=600)



#%% 

x = np.random.normal(0,1,10)
y = x + np.random.normal(0,1,10)
x = (x-x.mean())/x.std()
y = (y-y.mean())/y.std()

betas = np.polyfit(x,y,deg=1)
betas[0] = 0
plt.figure(figsize=(4,4))
for i in range(len(x)):
    plt.plot([x[i],x[i]], [y[i],(x*betas[0]+betas[1])[i]], color='red', linestyle='--', alpha=0.5, zorder=0)
plt.scatter(x,y,50, edgecolor='black')
plt.plot(x,x*betas[0]+betas[1], color='white')

betas = np.polyfit(x,y,deg=1)

cost = np.array([np.sum((y- (x*p+betas[1]))**2) for p in params])

plt.figure(figsize=(4,4))
plt.plot(params ,cost, marker = '.', markersize=25, markeredgecolor='black', color='white')
plt.xlabel(r"$\beta_1$")
plt.ylabel('error')






#%%

df = pd.DataFrame({
    'x1':x[:,0],
    'x2':x[:,1],
    'y':y
})

df.to_csv('./example_data.tsv', index= None)

#%%
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df = pd.DataFrame({
    'age': data.age,
    'blood_pressure': data.high_blood_pressure,
    'y': data.DEATH_EVENT
})

df.to_csv('./example_data_heart.tsv', index= None)


#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# HIER STEHEN DEINE DATEN (bereits in deinem Notebook definiert):
# x, y, params, betas

# 1. Berechne die Fehlerpunkte für DEINE 'params' und DEINE Daten (x, y)
# Wir nutzen genau deine Formel (x * p + betas[1]) für die Vorhersage
errors = []
for p in params:
    predictions = x * p + betas[1]
    mse = np.mean((predictions - y) ** 2)  # Mittlerer quadratischer Fehler
    errors.append(mse)

# 2. Figure mit zwei Subplots nebeneinander erstellen (1 Zeile, 2 Spalten)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Die Update-Funktion für das GIF
def update(frame_idx):
    ax1.clear()
    ax2.clear()
    
    p = params[frame_idx]
    current_error = errors[frame_idx]
    
    # --- Linker Plot: Exakt deine Plot-Logik mit deinen Daten ---
    ax1.scatter(x, y, 50, edgecolor='black', zorder=1)
    # Hinweis: Falls dein Hintergrund weiß ist, habe ich die Linie auf 'blue' gesetzt,
    # damit man sie sieht. Wenn dein Hintergrund dunkel ist, änder es wieder zu 'white'.
    ax1.plot(x, x * p + betas[1], color='white', zorder=2) 
    
    for i in range(len(x)):
        ax1.plot([x[i], x[i]], [y[i], (x * p + betas[1])[i]], color='red', linestyle='--', alpha=0.5, zorder=0)
        
    ax1.set_xlabel('x-Wert')
    ax1.set_ylabel('y-Wert')
    ax1.set_title("Daten mit 'Modell'")
    ax1.set_xlim(-3,3)
    ax1.set_ylim(-3,3)
    
    # --- Rechter Plot: Die dazu korrespondierende Fehlerfunktion ---
    # Zeichnet die Fehlerkurve basierend auf deinen params
    ax2.plot(params, errors, color='gray', linestyle='--', alpha=0.5)
    # Markiert den aktuellen Fehlerzustand mit einem roten Kreuz
    ax2.scatter(p, current_error, s=50, color='red', marker='X', zorder=2)
    
    ax2.set_xlabel('Parameter (Steigung)')
    ax2.set_ylabel('Error (MSE)')
    ax2.set_title('Fehlerfunktion')

# 3. Animation erstellen (nutzt die Länge deiner params)
ani = animation.FuncAnimation(fig, update, frames=len(params), interval=500)

# 4. Als GIF speichern
ani.save('optimierung_mit_fehlerfunktion.gif', writer='pillow',dpi=200)
plt.close()

print("GIF wurde erfolgreich mit deinen Daten gespeichert!")

#%%
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
cmap = ListedColormap(['#66c2a5', '#fc8d62'])
x,y = make_moons(n_samples=40, noise=0.1)
scaler = StandardScaler()
x = scaler.fit_transform(x)
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x,y);

x_mesh, y_mesh = np.meshgrid(np.arange(-3,3,0.1),np.arange(-3,3,0.1))
mesh = knn.predict(np.vstack((x_mesh.flatten(),y_mesh.flatten())).T).reshape(x_mesh.shape)


plt.figure(figsize=(4,4))
plt.scatter(x[:,0],x[:,1], c=y,cmap=cmap, edgecolors='black')
plt.ylabel('Merkmal 2')
plt.xlabel('Merkmal 1')
plt.tight_layout()
plt.savefig('./moons.png', dpi=600)

plt.figure(figsize=(4,4))
plt.contourf(x_mesh, y_mesh, mesh,1, cmap=cmap, alpha=0.2)
plt.scatter(x[:,0],x[:,1], c=y,cmap=cmap, edgecolors='black')
plt.ylabel('Merkmal 2')
plt.xlabel('Merkmal 1')
plt.tight_layout()
plt.savefig('./moons_decision.png', dpi=600)



#%%

np.random.seed(41)
cmap = ListedColormap(['#66c2a5', '#fc8d62'])
x,y = make_moons(n_samples=40, noise=2)
scaler = StandardScaler()
x = scaler.fit_transform(x)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y);
x_mesh, y_mesh = np.meshgrid(np.arange(-3,3,0.1),np.arange(-3,3,0.1))
mesh = knn.predict(np.vstack((x_mesh.flatten(),y_mesh.flatten())).T).reshape(x_mesh.shape)


plt.figure(figsize=(4,4))
plt.contourf(x_mesh, y_mesh, mesh,1, cmap=cmap, alpha=0.2)
plt.scatter(x[:,0],x[:,1], c=y,cmap=cmap, edgecolors='black')
plt.ylabel('Merkmal 2')
plt.xlabel('Merkmal 1')
plt.tight_layout()
plt.savefig('./moons_overfit.png', dpi=600)

np.random.seed(40)
x,y = make_moons(n_samples=40, noise=2)
x = scaler.fit_transform(x)
plt.figure(figsize=(4,4))
plt.contourf(x_mesh, y_mesh, mesh,1, cmap=cmap, alpha=0.2)
plt.scatter(x[:,0],x[:,1], c=y,cmap=cmap, edgecolors='black')
plt.ylabel('Merkmal 2')
plt.xlabel('Merkmal 1')
plt.tight_layout()
plt.savefig('./moons_newdata.png', dpi=600)

#%%

from sklearn.model_selection import cross_validate


cmap = ListedColormap(['#66c2a5', '#fc8d62'])
x,y = make_moons(n_samples=4000, noise=200)
x = scaler.fit_transform(x)
k_nei = np.arange(1,100)
test = []
train = []

for k in k_nei:
    knn = KNeighborsClassifier(n_neighbors=k)
    results = cross_validate(knn,x,y,return_train_score=True)

    test.append(results['test_score'].mean())
    train.append(results['train_score'].mean())

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.plot(k_nei, train, marker='.', markersize=20, label='Train data')
ax.plot(k_nei, test, marker='.', markersize=20, label='Test data')
ax.legend()
ax.xaxis.set_inverted(True)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Modell-Flexibilität')
plt.tight_layout()
fig.savefig('./train_test.png',dpi=600)