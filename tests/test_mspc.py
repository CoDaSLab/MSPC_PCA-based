from src.mspc_pca import mspc

import numpy as np

N = 100
M = 20
data = np.random.rand(N, M)
data[:, 0] = data[:, 0] + data[:, 2] * 2 # Introduce some correlation
data[:, 1] = data[:, 1] - data[:, 3] * 1.5
data[:,4] = data[:,4]*5

def test_U_square():
    mu, std, pca = mspc.train_params(data[:-10], 3, 2)
    data[-10:] *=2
    X_train = pca.fit_transform(data)
    d2 = mspc.U_squared(X_train, mu, std)

    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(d2)
    plt.show()











