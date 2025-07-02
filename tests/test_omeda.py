import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from mspc_pca import pca
from mspc_pca.omeda import omeda
from numpy.testing import assert_allclose

def test_omeda_with_anomalous_observation():
    # Simulate example data
    n_obs = 100
    n_vars = 10
    X = np.fromfunction(lambda i, j: (i * j + j**2 - i)/(i * j + 1), (n_obs, n_vars), dtype=int)

    # Introduce anomaly
    X[0, 0:2] = 1.25 * np.max(np.abs(X[:, 0:2]))
    X[0, -1] = -0.25 * np.max(np.abs(X[:, 0:2]))

    # PCA fit
    _, pca_model = pca.adjust_PCA(X, min(n_obs, n_vars), with_std=False)
    loadings = pca_model.components_.T

    # Mean centering
    m = np.mean(X, axis=0)
    Xcs = X - m

    # Dummy vector: mark first observation
    dummy = np.zeros((n_obs, 1))
    dummy[0] = 1

    # OMEDA
    omeda_vec = omeda(
        Xcs,
        dummy,
        loadings[:, :2],
        var_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "Z"],
        var_classes=["a", "a", "b", "b", "c", "b", "a", "c", "c", "b"],
        title="First obs vs. the others"
    )

    # Expected output vector (from known result)
    expected_vec = np.array([
        2.95883000e+04, 1.49990375e+04, 1.14570135e+01, 6.59048927e+01,
        2.22399328e+02, 5.62684483e+02, 1.19201213e+03, 2.23914884e+03,
        3.85637998e+03, -1.02095071e+03
    ])

    # Assertions
    assert omeda_vec is not None, "OMEDA result should not be None"
    assert isinstance(omeda_vec, np.ndarray), "OMEDA output should be a numpy array"
    assert omeda_vec.shape == expected_vec.shape, "Shape mismatch in OMEDA vector"
    
    # Check values are approximately equal
    assert_allclose(omeda_vec, expected_vec, rtol=1e-4, atol=1e-2)
