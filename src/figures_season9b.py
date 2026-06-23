import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

data = pd.read_csv('/home/philipp/Documents/GitHub/teaching/heart.csv')

features = data.columns[:-1]
x = data[features]
#x = x.apply(np.random.permutation)
y = data.target

pipe = Pipeline(
    [
        ('z_scaling', StandardScaler()),
        ('pca', PCA(n_components=0.90)),
        ('svc', SVC(kernel='linear',C=100))
        ]
)

cv  = cross_validate(pipe, x,y, cv=10, scoring='balanced_accuracy')
cv['test_score'].mean()




#%% build model 

svc = SVC(kernel='linear')
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
svc.fit(X=x_scaled,y=y);

y_predicted = svc.decision_function(x_scaled)

from scipy.stats import gaussian_kde


def cohensd(x, y):

    g0 = x[y == 0]
    g1 = x[y == 1]

    n1, n2 = len(g0), len(g1)
    
    var1, var2 = g0.var(), g1.var()

    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    diff = g0.mean() - g1.mean()
    
    return diff / pooled_sd

for i in features[:-1]:
    print(f'd for {i}: {cohensd(x[i],y)}')

cohensd(svc.decision_function(x_scaled),y)


sum(svc.decision_function(x_scaled[y==1])>1)/sum(svc.decision_function(x_scaled[y==0])>1)


#%% calculate custom linear regression

def logreg(x, params):
    y = np.ones(len(x))*params[0]
    for i in range(1,len(params)):
        y += params[i]*x[:,i-1]
    return 1/(1+np.exp(-y))

def cross_entropy(params, x,y, eps=1e-4):
    y_hat = logreg(x, params)

    return np.sum(np.log(1-y_hat+eps) * (1-y) + np.log(y_hat+eps)*(y))


from scipy.optimize import minimize

x,y = make_blobs(100, n_features= 2, centers = 2, cluster_std=5)
x_shuffled = np.array([np.random.permutation(x[:,i]) for i in range(x.shape[1])]).T

fit = minimize(cross_entropy, [1,1,1], args=(x,y))
fit_shuffled = minimize(cross_entropy, [1,1,1], args=(x_shuffled,y))

sum((np.array(fit.x[0] +  fit.x[1]*x[:,0] +fit.x[2]*x[:,1]) < 0) == y)/len(y)
sum((np.array(fit_shuffled.x[0] +  fit_shuffled.x[1]*x_shuffled[:,0] +fit_shuffled.x[2]*x_shuffled[:,1]) < 0) == y)/len(y)
plt.scatter(x[:,0], x[:,1], c=y)