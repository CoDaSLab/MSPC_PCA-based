from src.mspc_pca.pca import *
from src.mspc_pca import plot
import numpy as np

# Synthetic data for testing
np.random.seed(124) 
data = np.random.rand(50, 10) * 10 
data[:, 0] = data[:, 0] + data[:, 2] * 2 # Introduce some correlation
data[:, 1] = data[:, 1] - data[:, 3] * 1.5

obs_l = np.arange(50)
var_l = np.arange(10)

# Normalize the data
scaler = StandardScaler(with_std=False)
X = scaler.fit_transform(data)

# PCA model for testing
pca = PCA(n_components=5)
scores = pca.fit_transform(X)
loadings = pca.components_


def test_var_pca():

    print("Running test for var_pca function...")

    # 1. Generate some dummy data
    np.random.seed(123) 
    data = np.random.rand(50, 10) * 10 
    data[:, 0] = data[:, 0] + data[:, 2] * 2 # Introduce some correlation
    data[:, 1] = data[:, 1] - data[:, 3] * 1.5

    max_components_test = 5

    print("\nTest Case 1: Default parameters")
    fig, ax = plot.var_pca(data, max_components_test)
    plt.show()


    return

def test_scores():
    plot.scores(X, pca, 1, 2, labels=obs_l)
    plt.show()
    return

def test_loadings():
    plot.loadings(pca, 1, 2, labels=var_l)
    plt.show()
    return
    
def test_biplot():
    plot.biplot(X, pca, 1, 2, obs_l, var_l)
    plt.show(block=False)
    plot.biplot(X, pca, 3, 4, obs_l, var_l)
    plt.show()
    return