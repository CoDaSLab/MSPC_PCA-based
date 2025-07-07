import os, sys
sys.path.insert(1, os.getcwd())
import ckf
import pca

import numpy as np

n = 10
data = np.random.rand(n,n)


# pca.pca_resvar(data, n)

T, pca_model = pca.adjust_PCA(data, 2, with_std=True)

# pca.plot_scores_2d(T, 1, 2, pca_model)
# pca.plot_loadings_2d(1,2,pca_model)
pca.combine_plots(T,0,1,pca_model)