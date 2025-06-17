import numpy as np
import matplotlib.pyplot as plt

def ckf(xcs, T, P, plot=True):
    """
    CKF Algorithm: Adapted from Journal of Chemometrics, 29(8): 467-478, 2015
    Reference: https://github.com/josecamachop/MEDA-Toolbox/blob/master/toolbox/modelSel/ckf.m

    Computes the CKF cumulative press for a given dataset.
    @param xcs: Data matrix of shape [N x M] where N is the number of samples and M is the number of variables.
    @param T: Scores matrix of shape [N x A] where A is the number of components.
    @param P: Loadings matrix of shape [M x A] where A is the number of components.
    @param plot: If True, plots the CKF curve.
    @return: cumpress: np.ndarray Cumulative CKF values for each number of components.
    """
    N, M = xcs.shape
    A = T.shape[1]

    assert T.shape == (N, A), "T must be of shape [N x A]"
    assert P.shape == (M, A), "P must be of shape [M x A]"

    cumpress = np.zeros(A + 1)
    press = np.zeros((A + 1, M))

    for i in range(A + 1):
        if i > 0:
            p2 = P[:, :i]
            srec = T[:, :i] @ p2.T
            erec = xcs - srec
            term3p = erec
            term1p = xcs * (np.ones((N, 1)) @ np.sum(p2**2, axis=1, keepdims=True).T)
        else:
            term1p = np.zeros_like(xcs)
            term3p = xcs

        term1 = term1p**2
        term2 = 2 * term1p * term3p
        term3 = term3p**2

        press[i, :] = np.sum(term1 + term2 + term3, axis=0)
        cumpress[i] = np.sum(press[i, :])

    if plot:
        plt.figure()
        plt.plot(range(A + 1), cumpress / cumpress[0], marker='o')
        plt.xlabel("#PCs")
        plt.ylabel("ckf")
        plt.title("CKF Curve")
        plt.grid(True)
        plt.show()

    return cumpress
