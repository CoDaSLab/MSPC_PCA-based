from src.mspc_pca import mspc
from src.mspc_pca import plot
import matplotlib.pyplot as plt

import numpy as np

np.random.rand(124)
N = 100
M = 20
data = np.random.rand(N, M)
data[:, 0] = data[:, 0] + data[:, 2] * 2 # Introduce some correlation
data[:, 1] = data[:, 1] - data[:, 3] * 1.5
data[:,4] = data[:,4]*5

def test_U_square():
    k = 20
    mu, std, pca = mspc.train_params(data[:-k], 3, 2)
    data[-k:] *=1.5
    X = pca.fit_transform(data)
    U = mspc.U_squared(X, mu, std,)

    plot.plot_U_tt(U[:-k], U[-k:], [.10, .40], plot_train=True)

    plt.show()











