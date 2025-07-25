from src.mspc_pca.pca import *
from src.mspc_pca import plot
import numpy as np

# Synthetic data for testing
np.random.seed(124) 
data = np.random.rand(50, 10) * 10 
data[:, 0] = data[:, 0] + data[:, 2] * 2 # Introduce some correlation
data[:, 1] = data[:, 1] - data[:, 3] * 1.5

data[:,4] = data[:,4]*5

obs_l = [f'Obs_{i}' for i in range(50)] # Categorical observation labels
classes_num = np.arange(50).tolist() # Numerical classes for scores
classes_cat = [f'Class_{i%5}' for i in range(50)] # Categorical classes for scores

var_l = [f'Var_{i}' for i in range(10)] # Categorical variable labels
var_classes_num = np.arange(10).tolist() # Numerical classes for loadings
var_classes_cat = [f'Type_{i%3}' for i in range(10)] # Categorical classes for loadings

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
    _,_,scatter = plot.scores(X, pca, 1, 2, labels=obs_l, classes=classes_num, cmap='rainbow')
    plt.colorbar(scatter)
    plt.show()
    return

def test_loadings():
    _,_,scatter=plot.loadings(pca, 1, 2, labels=var_l, label_dist=0.01, classes=var_classes_num[:], cmap='rainbow')
    plt.colorbar(scatter)
    plt.show()
    return
    
def test_biplot_no_classes():
    """Test biplot with no classes."""
    _,ax, scatter = plot.biplot(X, pca, 1, 2, obs_l, var_l, size=50, loading_percentile=-.25)
    ax.set_title("Biplot: No Score & Loading Classes")
    plt.show()
    return

def test_biplot_num_num():
    """Test biplot with numerical score_classes and numerical loading_classes."""
    fig, ax, scatter = plot.biplot(X, pca, 1, 2, 
                                score_labels=obs_l, 
                                loading_labels=var_l, 
                                size=50, 
                                score_classes=classes_num, 
                                loading_classes=var_classes_num,
                                loading_percentile=-.25)
    ax.set_title("Biplot: Numerical Score & Loading Classes")
    plt.show()

def test_biplot_cat_cat():
    """Test biplot with categorical score_classes and categorical loading_classes."""
    fig, ax, scatter = plot.biplot(X, pca, 1, 2, 
                                score_labels=obs_l, 
                                loading_labels=var_l, 
                                size=50, 
                                score_classes=classes_cat, 
                                loading_classes=var_classes_cat,
                                loading_percentile=+.25)
    ax.set_title("Biplot: Categorical Score & Loading Classes")
    plt.show()

def test_biplot_num_cat():
    """Test biplot with numerical score_classes and categorical loading_classes."""
    fig, ax, scatter = plot.biplot(X, pca, 1, 2, 
                                score_labels=obs_l, 
                                loading_labels=var_l, 
                                size=50, 
                                score_classes=classes_num, 
                                loading_classes=var_classes_cat,
                                loading_cmap="rainbow",
                                loading_percentile=-.25)
    ax.set_title("Biplot: Numerical Score Classes, Categorical Loading Classes")
    legend = ax.get_legend()
    legend.set_loc("lower left")
    plt.show()

def test_biplot_cat_num():
    """Test biplot with categorical score_classes and numerical loading_classes."""
    fig, ax, scatter = plot.biplot(X, pca, 1, 2, 
                                score_labels=obs_l, 
                                loading_labels=var_l, 
                                size=50, 
                                score_classes=classes_cat, 
                                loading_classes=var_classes_num,
                                loading_percentile=-.25)
    ax.set_title("Biplot: Categorical Score Classes, Numerical Loading Classes")
    plt.show()


def test_biplot_preexisting_axis():
    """Test biplot with numerical score_classes and numerical loading_classes."""
    fig, axes = plt.subplots(2,1, figsize=(10, 6))
    fig, ax, scatter = plot.biplot(X, pca, 1, 2, 
                                score_labels=obs_l, 
                                loading_labels=var_l, 
                                size=50, 
                                score_classes=classes_num, 
                                loading_classes=var_classes_num,
                                loading_percentile=-.25,
                                ax = axes[0])
    ax.set_title("Biplot: Numerical Score & Loading Classes")

    fig, ax2, scatter = plot.biplot(X, pca, 1, 2, 
                                score_labels=obs_l, 
                                loading_labels=var_l, 
                                size=50, 
                                score_classes=classes_cat, 
                                loading_classes=var_classes_cat,
                                loading_percentile=-.25,
                                ax = axes[1])
    ax2.set_title("Biplot: Categorical Score & Loading Classes")
    plt.tight_layout()
    plt.show()