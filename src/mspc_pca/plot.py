import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
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

def biplot(data, pca_model, pc1: int, pc2: int,
           score_labels: list = None,
           loading_labels: list = None,
           label_dist: float = 1.0,
           score_classes: list = None,
           loading_classes: list = None,
           score_cmap: str = 'viridis',
           loading_cmap: str = 'coolwarm',
           size: int = 20,
           loading_percentile:float = 0.10,
           ax=None):
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
    pc1, pc2 = pc1 - 1, pc2 - 1  # Indexación 0-based

    score_labels = np.array(score_labels)
    loading_labels = np.array(loading_labels)

    scores = pca_model.fit_transform(data)
    explained_variance = pca_model.explained_variance_ratio_ * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    scores_scaled = scores / np.max(np.abs(scores), axis=0)
    loadings_scaled = pca_model.components_ / np.max(np.abs(pca_model.components_), axis=1)[:, np.newaxis]
    loads_dist = (loadings_scaled[pc1,:]**2+loadings_scaled[pc2,:]**2)
    max_load = np.max(loads_dist)
    threshold_dist = loading_percentile * max_load
    if threshold_dist >= 0:
        mask = loads_dist >= threshold_dist
    else:
        mask = loads_dist <= np.abs(threshold_dist)

    loadings_scaled_masked = loadings_scaled[:,mask]

    # Color loadings
    if loading_classes is not None:
        loading_classes_masked = np.array(loading_classes)[mask]
        if np.issubdtype(np.array(loading_classes_masked).dtype, np.number):
            loading_classes_type = 'numeric'
            # Numerical classes: Normalize and map to colormap
            loading_norm = plt.Normalize(vmin=np.min(loading_classes_masked), vmax=np.max(loading_classes_masked))
            loading_colors = plt.get_cmap(loading_cmap)(loading_norm(loading_classes_masked))
        else:
            # Categorical classes
            loading_classes_type = 'categorical'
            unique_loading_classes = np.unique(loading_classes_masked)
            num_unique_loading_classes = len(unique_loading_classes)
            cmap = plt.get_cmap(loading_cmap, num_unique_loading_classes)
            category_colors = {cls: cmap(i) for i, cls in enumerate(unique_loading_classes)}
            
            # Assign colors to each loading based on its class
            loading_colors = np.array([category_colors[cls] for cls in loading_classes_masked])
    else:
        loading_classes_type = 'categorical'
        loading_classes_masked = ['Loadings'] * loadings_scaled_masked.shape[1]
        unique_loading_classes = ['Loadings']
        # Default to red if no loading_classes are provided
        loading_colors = ['red'] * loadings_scaled_masked.shape[1]
        loading_colors = to_rgba_array(loading_colors)

    # Loadings arrows
    for i in range(loadings_scaled_masked.shape[1]):
        ax.annotate(
            '',
            xy=(loadings_scaled_masked[pc1, i], loadings_scaled_masked[pc2, i]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->',
                            color=loading_colors[i],
                            linewidth=1,
                            alpha=0.7)
        )

    # Loading labels
    if loading_labels is not None:
        filtered = filter_labels(loadings_scaled_masked.T, loading_labels[mask], pc1, pc2, min_dist=label_dist)
        for x, y, label in filtered:
            ax.text(x, y, label, fontsize=8, color='grey')

    # Score colors
    if score_classes is not None:
        if np.issubdtype(np.array(score_classes).dtype, np.number):
            score_classes_type = 'numeric'
            # Numerical classes: Normalize and map to colormap
            score_norm = plt.Normalize(vmin=np.min(score_classes), vmax=np.max(score_classes))
            score_colors = plt.get_cmap(score_cmap)(score_norm(score_classes))
        else:
            # Categorical classes
            score_classes_type = 'categorical'
            unique_score_classes = np.unique(score_classes)
            num_unique_score_classes = len(unique_score_classes)
            cmap = plt.get_cmap(score_cmap, num_unique_score_classes)
            category_colors = {cls: cmap(i) for i, cls in enumerate(unique_score_classes)}
            
            # Assign colors to each score based on its class
            score_colors = np.array([category_colors[cls] for cls in score_classes])
    else:
        score_classes_type = 'categorical'
        # Default to blue if no score_classes are provided
        score_classes = ['Scores'] * scores_scaled.shape[0]
        unique_score_classes = ['Scores']
        score_colors = ['blue'] * scores_scaled.shape[0]
        score_colors = to_rgba_array(score_colors)
        
    # Plot scores
    if score_classes_type == 'numeric':
        scatter = ax.scatter(scores_scaled[:, pc1], scores_scaled[:, pc2],
                                c=score_colors, alpha=0.9, s=size, zorder=3)
        # Colorbar
        sm = ScalarMappable(cmap=plt.get_cmap(score_cmap), norm=score_norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.075)
        cbar.set_label('Observation class')

    elif score_classes_type == 'categorical':
        for i, cls in enumerate(unique_score_classes):
            class_idx = (np.array(score_classes) == cls)
            scatter = ax.scatter(scores_scaled[class_idx, pc1], scores_scaled[class_idx, pc2],
                                c=score_colors[class_idx], label=cls, alpha=0.9, s=size, zorder=3)

    # Scores labels
    if score_labels is not None:
        filtered = filter_labels(scores_scaled, score_labels, pc1, pc2, min_dist=label_dist)
        for x, y, label in filtered:
            offset = 0.025
            sign_x = x / np.abs(x) if x != 0 else 1
            sign_y = y / np.abs(y) if y != 0 else 1
            ax.text(x + offset * sign_x, y + offset * sign_y, label,
                    fontsize=9, color='black', ha='center', va='center')

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(f'PC{pc1+1} ({explained_variance[pc1]:.2f}%)')
    ax.set_ylabel(f'PC{pc2+1} ({explained_variance[pc2]:.2f}%)')

    # Limits
    score_coords = np.column_stack((scores_scaled[:, pc1], scores_scaled[:, pc2]))
    loading_coords = np.column_stack((loadings_scaled_masked[pc1], loadings_scaled_masked[pc2]))
    all_coords = np.vstack((score_coords, loading_coords))
    ax.update_datalim(all_coords)
    ax.autoscale_view()
    
    # Arrows in the legend
    arrow_legends = []
    handles, labels = ax.get_legend_handles_labels()
    if loading_classes_type == 'numeric':
        sm2 = ScalarMappable(cmap=plt.get_cmap(loading_cmap), norm=loading_norm)

        # Add colorbar for loadings
        if score_classes_type == 'numeric':
            divider2 = make_axes_locatable(ax)
            cax2 = divider2.append_axes("right", size='5%', pad=0.1)
            cbar2 = fig.colorbar(sm2, cax=cax2, orientation='vertical')
        else:
            cbar2 = fig.colorbar(sm2, ax=ax, orientation='vertical', pad=0.075)
        
        cbar2.set_label('Variable class')

        # Add loadings classes to legend
        arrow_legends.append(Line2D([0], [0], color='grey', lw=1, marker='>', markersize=6, label='Loadings (as arrows)'))
        labels.append('Loadings')
    elif loading_classes_type == 'categorical':
        for i, cls in enumerate(unique_loading_classes):
            class_idx = (np.array(loading_classes_masked) == cls)
            arrow_legends.append(Line2D([0], [0], color=loading_colors[class_idx, :][0], lw=1, marker='>', markersize=6, label=cls))
            labels.append(cls)
            
    handles.extend(arrow_legends)
    
    ax.legend(handles=handles, labels=labels, loc="upper right")

    return fig, ax, scatter