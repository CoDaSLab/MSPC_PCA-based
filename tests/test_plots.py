from src.mspc_pca.pca import *
from src.mspc_pca import plot
import numpy as np

# Synthetic data for testing
np.random.seed(124) 
data = np.random.rand(50, 10) * 10 
data[:, 0] = data[:, 0] + data[:, 2] * 2 # Introduce some correlation
data[:, 1] = data[:, 1] - data[:, 3] * 1.5

data[:,4] = data[:,4]*5

obs_l = np.arange(50)
classes = np.arange(50)
var_l = np.arange(10)

# Normalize the data
scaler = StandardScaler(with_std=False)
X = scaler.fit_transform(data)

# PCA model for testing
pca = PCA(n_components=5)
scores = pca.fit_transform(X)
loadings = pca.components_


def test_var_pca():
    max_components_test = 5
    plot.var_pca(data, max_components_test)
    plt.show()
    return

def test_scores():
    _,_,scatter = plot.scores(X, pca, 1, 2, labels=obs_l, classes=classes, cmap='rainbow')
    plt.colorbar(scatter, )
    plt.show()
    return

def test_loadings():
    _,_,scatter=plot.loadings(pca, 1, 2, labels=var_l, label_dist=0.01, classes=var_l[:], cmap='rainbow')
    plt.colorbar(scatter, )
    plt.show()
    return
    
def test_biplot():
    _,_, scatter = plot.biplot(X, pca, 1, 2, obs_l, var_l, size=50, score_classes=classes, loading_classes=var_l)
    plt.colorbar(scatter, )
    plt.show()
    return
