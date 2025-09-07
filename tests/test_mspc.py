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

    fig, ax = plt.subplots(1,1, figsize=(10,10))

    plot.plot_U_tt(U[:-k], U[-k:], [.10, .40], plot_train=True, ax = ax)

    plt.show()


def test_tscore_with_array():
    X = np.random.randn(20, 5)
    result = mspc.tscore(X, weight=0.5, norm_quantile=0.95, n_components=2)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == X.shape[0]
    assert not np.any(np.isnan(result))


def test_tscore_with_tuple():
    D = np.random.rand(20)
    Q = np.random.rand(20)
    result = mspc.tscore((D, Q), weight=0.7, norm_quantile=0.9, n_components=2)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(D)
    assert not np.any(np.isnan(result))


def test_tscore_tt_with_arrays():
    X_train = np.random.randn(30, 4)
    X_test = np.random.randn(10, 4)
    T_train, T_test = mspc.tscore_tt(X_train, X_test, weight=0.6, norm_quantile=0.95, n_components=2)
    assert isinstance(T_train, np.ndarray)
    assert isinstance(T_test, np.ndarray)
    assert T_train.shape[0] == X_train.shape[0]
    assert T_test.shape[0] == X_test.shape[0]
    assert not np.any(np.isnan(T_train))
    assert not np.any(np.isnan(T_test))


def test_tscore_tt_with_tuples():
    D_train = np.random.rand(30)
    Q_train = np.random.rand(30)
    D_test = np.random.rand(10)
    Q_test = np.random.rand(10)
    T_train, T_test = mspc.tscore_tt((D_train, Q_train), (D_test, Q_test), weight=0.4, norm_quantile=0.9, n_components=2)
    assert isinstance(T_train, np.ndarray)
    assert isinstance(T_test, np.ndarray)
    assert T_train.shape[0] == len(D_train)
    assert T_test.shape[0] == len(D_test)