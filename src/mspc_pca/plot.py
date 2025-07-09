import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from .ckf import ckf
from matplotlib.lines import Line2D 


def var_pca(data, max_components, with_std=True, with_ckf=True, exclude_zero=False, ax=None):
    """
    Performs PCA and plots the residual variance and the CKF curve to select the optimal number of components using the elbow method.
    
    @param data: Data matrix (windows x frequencies)
    @param max_components: Number of principal components to consider
    @param with_std: If True, normalizes the data before PCA
    @param with_ckf: If True, calculates the CKF curve
    @param exclude_zero: If True, excludes variance values associated with 0 components. This is useful for visualization but may affect "elbow" detection.
    @param ax: Optional matplotlib Axes object to plot on. If None, a new Figure and Axes will be created.
    @return: A tuple containing the matplotlib Figure (or None if 'ax' was provided) and Axes objects.
    """
    scaler = StandardScaler(with_std=with_std)
    data_norm = scaler.fit_transform(data)

    pca = PCA(n_components=max_components)
    data_fitted = pca.fit_transform(data_norm)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    if not exclude_zero:
        cumulative_explained_variance = np.insert(cumulative_explained_variance, 0, 0)
    residual_variance = 1 - cumulative_explained_variance

    if with_ckf:
        scores = data_fitted
        loadings = pca.components_[:max_components, :]
        ckf_cumpress = ckf(data_norm, scores, loadings.T, plot=False)
        if not exclude_zero:
            ckf_cumpress = ckf_cumpress / ckf_cumpress[0]  # Normalize CKF
        else:
            ckf_cumpress = ckf_cumpress[1:] / ckf_cumpress[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else: fig = None 

    plot_range = range(1, max_components + 1) if exclude_zero else range(0, max_components + 1)
    ax.plot(plot_range, residual_variance, label='Residual Variance', color='red', marker='o')
    ax.set_xlabel('Number of Principal Components', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('Residual Variance', fontsize=20)
    
    if with_ckf:
        ax.set_title('Residual Variance Curve and CKF Curve')
        ax.plot(plot_range, ckf_cumpress, label='CKF Curve', color='blue', marker='s')
    else:
        ax.set_title('Residual Variance Curve')


    ax.legend()
    
    return fig, ax


def filter_labels(data, labels, pc1, pc2, min_dist=0.05):
    """
    Filters out labels that are too close to each other based on a minimum distance.

    :param data: Data matrix of shape (n_samples, n_features).
    :param labels: List of labels corresponding to the data points.
    :param pc1: Index of the first principal component (0-based).
    :param pc2: Index of the second principal component (0-based).
    :param min_dist: Minimum allowed distance between labeled points.
    :return: List of filtered (x, y, label) tuples.
    """
    coords = data[:, [pc1, pc2]]
    kept = []
    added = []

    for i, (x, y) in enumerate(coords):
        point = np.array([[x, y]])
        if len(added) == 0:
            kept.append((x, y, labels[i]))
            added.append(point)
        else:
            dists = pairwise_distances(point, np.vstack(added))
            if np.min(dists) >= min_dist:
                kept.append((x, y, labels[i]))
                added.append(point)

    return kept

def scores(data, pca_model, pc1: int, pc2: int, labels: list = None, 
           label_dist: float = 1.0, classes: list = None, cmap:str='viridis', ax=None):
    """
    Plots the 2D PCA scores with optional point labeling and class-based coloring.
    
    :param data: Original data to plot as scores.
    :param pca_model: Fitted PCA model (sklearn.decomposition.PCA object).
    :param pc1: Index of the first principal component to plot on the x-axis (1-based).
    :param pc2: Index of the second principal component to plot on the y-axis (1-based).
    :param labels: Optional list of text labels for score points.
    :param label_dist: Minimum distance between labeled points.
    :param classes: Optional list or array of class labels for coloring the points.
    :param ax: Optional matplotlib Axes object. If None, a new Figure and Axes will be created.
    :return: A tuple (Figure, Axes). Figure is None if ax was provided.
    """
    pc1, pc2 = pc1 - 1, pc2 - 1  # Convert to 0-based indexing

    scores = pca_model.fit_transform(data)
    explained_variance = pca_model.explained_variance_ratio_ * 100  # Convert to percentage

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None

    # Plot with color if classes are provided
    if classes is not None:
        scatter = ax.scatter(scores[:, pc1], scores[:, pc2], c=classes, cmap=cmap, alpha=0.9)

    else:
        ax.scatter(scores[:, pc1], scores[:, pc2], alpha=0.9)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(f'PC{pc1+1} ({explained_variance[pc1]:.2f}%)', fontsize=20)
    ax.set_ylabel(f'PC{pc2+1} ({explained_variance[pc2]:.2f}%)', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title('2D Scores', fontsize=16)

    if labels is not None:
        filtered = filter_labels(scores, labels, pc1, pc2, min_dist=label_dist)
        for x, y, label in filtered:
            ax.text(x, y, label, fontsize=8, color='black')

    return fig, ax, scatter

def loadings(pca_model, pc1: int, pc2: int, labels: list = None, label_dist:float = 0.01, classes: list = None, cmap:str='viridis', ax=None):
    """
    Plots the 2D loadings for the principal components.
    
    :param pca_model: Fitted PCA model (sklearn.decomposition.PCA object).
    :param pc1: Index of the first principal component to plot on the x-axis (1-based).
    :param pc2: Index of the second principal component to plot on the y-axis (1-based).
    :param labels: Optional list of labels for the loading points (original features).
    :param ax: Optional matplotlib Axes object to plot on. If None, a new Figure and Axes will be created.
    :return: A tuple containing the matplotlib Figure (or None if 'ax' was provided) and Axes objects.
    """
    pc1, pc2 = pc1-1, pc2-1 # Adjust to 0-based indexing for Python
    
    explained_variance = pca_model.explained_variance_ratio_ * 100  # Convert to percentage

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None 

    # Plot loadings as points
    loadings = pca_model.components_
    # ax.scatter(loadings[pc1], loadings[pc2], alpha=0.9)
    
    if labels is not None:
        filtered = filter_labels(loadings, labels, pc1, pc2, min_dist=label_dist)
        for x, y, label in filtered:
            ax.text(x, y, label, fontsize=8, color='black')

    # Plot with color if classes are provided
    if classes is not None:
        scatter = ax.scatter(loadings[pc1], loadings[pc2], c=classes, cmap=cmap, alpha=0.9)

    else:
        ax.scatter(loadings[pc1], loadings[pc2], alpha=0.9)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Horizontal axis
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)  # Vertical axis
    ax.set_xlabel(f'PC{pc1+1} ({explained_variance[pc1]:.2f}%)', fontsize=20)
    ax.set_ylabel(f'PC{pc2+1} ({explained_variance[pc2]:.2f}%)', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title('2D Loadings', fontsize = 16)
    
    return fig, ax, scatter

def biplot(data, pca_model, pc1: int, pc2: int, score_labels: list = None, loading_labels: list = None, label_dist:float = 1.0, score_classes: list = None, cmap:str='viridis', size:int=20, ax=None):
    """
    Combines score and loading plots into a single superimposed graph,
    scaling both scores and loadings to maintain their relative positions.
    
    :param data: data to transform into scores
    :param pca_model: Fitted PCA model (sklearn.decomposition.PCA object) used to generate the scores and loadings.
    :param pc1: Index of the first principal component to plot on the x-axis (1-based).
    :param pc2: Index of the second principal component to plot on the y-axis (1-based).
    :param score_labels: Optional list of labels for the score points (observations).
    :param loading_labels: Optional list of labels for the loading points (original features).
    :param ax: Optional matplotlib Axes object to plot on. If None, a new Figure and Axes will be created.
    :return: A tuple containing the matplotlib Figure (or None if 'ax' was provided) and Axes objects.
    """
    pc1, pc2 = pc1-1, pc2-1 # Adjust to 0-based indexing for Python

    scores = pca_model.fit_transform(data)
    explained_variance = pca_model.explained_variance_ratio_ * 100  # Convert to percentage

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8)) # Increased figsize for better visibility
    else:
        fig = None

    scores_scaled = scores / np.max(np.abs(scores), axis=0)
    loadings_scaled = pca_model.components_ / np.max(np.abs(pca_model.components_), axis=1)[:, np.newaxis]

    # Plot loadings
    # ax.scatter(loadings_scaled[pc1], loadings_scaled[pc2], alpha=0.5, label='Loadings', color='red')
    if loading_labels is not None:
        filtered = filter_labels(loadings_scaled.T, loading_labels, pc1, pc2, min_dist=label_dist)
        for x, y, label in filtered:
            ax.text(x, y, label, fontsize=8, color='grey')
    
    for i in range(loadings_scaled.shape[1]):
        ax.annotate(
            '',  # No text
            xy=(loadings_scaled[pc1, i], loadings_scaled[pc2, i]),  # Arrow tip (target)
            xytext=(0, 0),  # Arrow start (origin)
            arrowprops=dict(arrowstyle='->', color='red', linewidth=1, alpha=0.5)
        )


    # Plot scores
    # ax.scatter(scores_scaled[:, pc1], scores_scaled[:, pc2], alpha=0.9, label='Scores', color='blue', s=size,zorder=3) # Increased marker size
    # Plot with color if classes are provided
    if score_classes is not None:
        scatter = ax.scatter(scores_scaled[:, pc1], scores_scaled[:, pc2], c=score_classes, cmap=cmap, label ='Scores',alpha=0.9, s=size,zorder=3)
    else:
        scatter = ax.scatter(scores_scaled[:, pc1], scores_scaled[:, pc2], c = 'blue', label ='Scores',alpha=0.9, s=size,zorder=3)

    if score_labels is not None:
        filtered = filter_labels(scores_scaled, score_labels, pc1, pc2, min_dist=label_dist)
        for x, y, label in filtered:
            offset = 0.025
            sign_x = x/np.abs(x)
            sign_y = y/np.abs(y)
            ax.text(x + offset*sign_x, y + offset*sign_y, label, 
                    fontsize=9, color='black', ha='center', va='center')
            

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

    ax.set_xlabel(f'PC{pc1+1} ({explained_variance[pc1]:.2f}%)')
    ax.set_ylabel(f'PC{pc2+1} ({explained_variance[pc2]:.2f}%)')
    # Coordenadas de scores y loadings
    score_coords = np.column_stack((scores_scaled[:, pc1], scores_scaled[:, pc2]))
    loading_coords = np.column_stack((loadings_scaled[pc1], loadings_scaled[pc2]))

    # Actualizar límites con ambos conjuntos
    all_coords = np.vstack((score_coords, loading_coords))
    ax.update_datalim(all_coords)
    ax.autoscale_view()

    arrow_legend = Line2D([0], [0], color='red', lw=1, marker='>', markersize=6, label='Loadings (as arrows)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(arrow_legend)
    labels.append('Loadings')

    ax.legend(handles=handles, labels=labels)
    
    return fig, ax, scatter