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
           markers: list = None,
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
            ax.text(x, y, label, fontsize=9, color="#242424")

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
        
    # Set markers
    if markers is None:
        markers = ['o'] * scores_scaled.shape[0]
    unique_markers = np.unique(markers)

    # Plot scores
    if score_classes_type == 'numeric':
        for marker in unique_markers:
            marker_idx = (np.array(markers) == marker)
            scatter = ax.scatter(scores_scaled[marker_idx, pc1], scores_scaled[marker_idx, pc2],
                                    c=score_colors[marker_idx], alpha=0.9, s=size, marker=marker, zorder=3)
        # Colorbar
        sm = ScalarMappable(cmap=plt.get_cmap(score_cmap), norm=score_norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.075)
        cbar.set_label('Observation class')

    elif score_classes_type == 'categorical':
        for cls in unique_score_classes:
            class_idx = (np.array(score_classes) == cls)
            plot_label = True
            for marker in unique_markers:
                marker_idx = (np.array(markers) == marker) & class_idx
                if np.any(marker_idx):
                    scatter = ax.scatter(scores_scaled[marker_idx, pc1], scores_scaled[marker_idx, pc2],
                                        c=score_colors[marker_idx], label=cls if plot_label else None,
                                        alpha=0.9, s=size, marker=marker, zorder=3)
                    plot_label = False

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

    elif loading_classes_type == 'categorical':
        for i, cls in enumerate(unique_loading_classes):
            class_idx = (np.array(loading_classes_masked) == cls)
            arrow_legends.append(Line2D([0], [0], color=loading_colors[class_idx, :][0], lw=1, marker='>', markersize=6, label=cls))
            labels.append(cls)
            
    handles.extend(arrow_legends)
    if len(handles) > 0:
        ax.legend(handles=handles, labels=labels, loc="upper right")

    return fig, ax, scatter

def plot_DQ(D, Q, threshold_D, threshold_Q, logscale=False, event_index=None, 
             alpha=None, type_q='Jackson', labels=None, opacity=None, ax=None):
    """
    Plots D-statistic and Q-statistic values.
    Highlights bars associated with `event_index` in red.

    :param D: Vector of D values.
    :param Q: Vector of Q values.
    :param threshold_D: Threshold(s) for D-statistic. Can be a scalar or a list.
    :param threshold_Q: Threshold(s) for Q-statistic. Can be a scalar or a list.
    :param logscale: If True, apply a logarithmic scale to the y-axis. Defaults to False.
    :param event_index: Optional list/array of indices to highlight in red. Defaults to None.
    :param alpha: Optional list/array of alpha values corresponding to thresholds, used for labels.
                  Defaults to None.
    :param type_q: Type of Q calculation, used in Q threshold label. Defaults to 'Jackson'.
    :param labels: Optional list of strings for x-axis tick labels. Defaults to None.
    :param opacity: Optional list of opacity values for each bar. Defaults to None.
    :param ax: Optional list or tuple of two matplotlib Axes objects ([ax_D, ax_Q]) to plot on.
               If None, a new Figure and two Axes will be created. Defaults to None.
    :return: A tuple (Figure, list_of_Axes).
    """
    if isinstance(event_index, int):
        event_index = [event_index]

    fig = None # Initialize fig to None, will be set if a new figure is created

    if ax is None:
        # Create a new figure and two subplots if no axes are provided
        fig, (ax_d, ax_q) = plt.subplots(2, 1, figsize=(12, 8))
    elif isinstance(ax, (list, tuple)) and len(ax) == 2:
        # Use provided axes
        ax_d, ax_q = ax[0], ax[1]
        fig = ax_d.figure # Get the figure from the provided axes
    else:
        raise ValueError("If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects.")

    n_data = len(D)
    x_indices = np.arange(n_data)

    # --- Plotting D ---
    # Determine colors for D plot
    if event_index is not None:
        colors_d = ['red' if i in event_index else 'blue' for i in range(n_data)]
    else:
        colors_d = 'blue'

    # Add opacity to colors
    if opacity is not None and len(opacity) == len(D):
        colors_d = to_rgba_array(colors_d, alpha=opacity)
    
    ax_d.bar(x_indices, D, color=colors_d, label='D')

    # Handle thresholds for D
    if np.isscalar(threshold_D):
        ax_d.axhline(y=threshold_D, linestyle='--', label='D threshold', color='red')
    else:
        threshold_D = np.atleast_1d(threshold_D)
        alpha_D = np.atleast_1d(alpha) if alpha is not None else [None] * len(threshold_D)
        for i, th in enumerate(threshold_D):
            label = f'D threshold $\\alpha$={alpha_D[i]}' if alpha_D[i] is not None else 'D threshold'
            ax_d.axhline(y=th, linestyle='--', label=label, color='red')

    ax_d.set_title("D-statistic")
    ax_d.set_ylabel('D')
    if logscale:
        ax_d.set_yscale('log')
    ax_d.legend(loc='upper left')
    ax_d.grid(True)

    # --- Plotting Q ---
    # Determine colors for Q plot
    if event_index is not None:
        colors_q = ['red' if i in event_index else 'green' for i in range(n_data)]
    else:
        colors_q = 'green'

    # Add opacity to colors
    if opacity is not None and len(opacity) == len(Q):
        colors_q = to_rgba_array(colors_q, alpha=opacity)

    ax_q.bar(x_indices, Q, color=colors_q, label='Q')

    # Handle thresholds for Q
    if np.isscalar(threshold_Q):
        ax_q.axhline(y=threshold_Q, linestyle='--', label='Q threshold', color='red')
    else:
        threshold_Q = np.atleast_1d(threshold_Q)
        alpha_Q = np.atleast_1d(alpha) if alpha is not None else [None] * len(threshold_Q)
        for i, th in enumerate(threshold_Q):
            label = f'Q threshold ({type_q}) $\\alpha$={alpha_Q[i]}' if alpha_Q[i] is not None else 'Q threshold'
            ax_q.axhline(y=th, linestyle='--', label=label, color='red')

    ax_q.set_title('Q-statistic')
    ax_q.set_ylabel('Q')
    if logscale:
        ax_q.set_yscale('log')
    ax_q.legend(loc='upper left')
    ax_q.grid(True)

    # Apply x-axis labels if provided
    if labels is not None:
        if len(labels) != n_data:
            print(f"Warning: Length of 'labels' ({len(labels)}) does not match the number of plotted data points ({n_data}). Labels will not be applied.")
        else:
            ax_d.set_xticks(x_indices)
            ax_d.set_xticklabels(labels, rotation=45, ha='right')
            ax_q.set_xticks(x_indices)
            ax_q.set_xticklabels(labels, rotation=45, ha='right')
            ax_q.tick_params(axis='x', which='major', pad=10)

    return fig, [ax_d, ax_q]


def plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, 
                alpha=None, type_q='Jackson', logscale=False, event_index=None, 
                labels=None, plot_train=True, opacity=None, ax=None):
    """
    Plots the D-statistic and Q-statistic results for
    training and test datasets, supporting one or multiple thresholds.
    Highlights bars associated with `event_index` in red.

    :param D_train: D-statistic for the training data.
    :param Q_train: Q-statistic values for the training data.
    :param D_test: D-statistic values for the test data.
    :param Q_test: Q-statistic values for the test data.
    :param threshold_D: Threshold(s) for D-statistic. Can be a scalar or a list.
    :param threshold_Q: Threshold(s) for Q-statistic. Can be a scalar or a list.
    :param alpha: Optional list/array of alpha values corresponding to thresholds, used for labels.
                  Defaults to None.
    :param type_q: Type of Q calculation, used in Q threshold label. Defaults to 'Jackson'.
    :param logscale: If True, apply a logarithmic scale to the y-axis. Defaults to False.
    :param event_index: Optional list/array of indices to highlight in red. Defaults to None.
    :param labels: Optional list of strings for x-axis tick labels. Defaults to None.
    :param plot_train: If True, both training and test data are plotted. If False, only
                       test data is plotted. Defaults to True.
    :param opacity: List of opacity values for each bar. Defaults to None.
    :param ax: Optional list or tuple of two matplotlib Axes objects ([ax_D, ax_Q]) to plot on.
               If None, a new Figure and two Axes will be created. Defaults to None.
    :return: A tuple (Figure, list_of_Axes).
    """
    if isinstance(event_index, int):
        event_index = [event_index]
        
    fig = None # Initialize fig to None, will be set if a new figure is created

    if ax is None:
        # Create a new figure and two subplots if no axes are provided
        fig, (ax_d, ax_q) = plt.subplots(2, 1, figsize=(12, 8))
    elif isinstance(ax, (list, tuple)) and len(ax) == 2:
        # Use provided axes
        ax_d, ax_q = ax[0], ax[1]
        fig = ax_d.figure # Get the figure from the provided axes
    else:
        raise ValueError("If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects.")

    n_train = len(D_train)
    n_test = len(D_test)

    if plot_train:
        # Concatenate train and test data for plotting
        D_to_plot = np.concatenate([D_train, D_test])
        Q_to_plot = np.concatenate([Q_train, Q_test])
        n_data_to_plot = len(D_to_plot)
        x_indices = np.arange(n_data_to_plot)

        # Determine colors for D plot
        if event_index is not None:
            colors_d = ['red' if i in event_index else ('blue' if i < n_train else 'orange') for i in range(n_data_to_plot)]
        else:
            colors_d = ['blue' if i < n_train else 'orange' for i in range(n_data_to_plot)]
        
        # Add opacity to colors
        if opacity is not None and len(opacity) == len(colors_d):
            colors_d = to_rgba_array(colors_d, alpha=opacity)

        # Plot D_train and D_test
        ax_d.bar(np.arange(n_train), D_train, color=colors_d[:n_train], label='Train')
        ax_d.bar(np.arange(n_train, n_data_to_plot), D_test, color=colors_d[n_train:], label='Test')
        ax_d.axvline(x=n_train - 0.5, color='black', linestyle='--', label='Train/Test Split')

        # Determine colors for Q plot
        if event_index is not None:
            colors_q = ['red' if i in event_index else ('green' if i < n_train else 'orange') for i in range(n_data_to_plot)]
        else:
            colors_q = ['green' if i < n_train else 'orange' for i in range(n_data_to_plot)]

        # Add opacity to colors
        if opacity is not None and len(opacity) == len(colors_q):
            colors_q = to_rgba_array(colors_q, alpha=opacity)

        # Plot Q_train and Q_test
        ax_q.bar(np.arange(n_train), Q_train, color=colors_q[:n_train], label='Train')
        ax_q.bar(np.arange(n_train, n_data_to_plot), Q_test, color=colors_q[n_train:], label='Test')
        ax_q.axvline(x=n_train - 0.5, color='black', linestyle='--', label='Train/Test Split')

    else: # Only plot test data
        D_to_plot = D_test
        Q_to_plot = Q_test
        n_data_to_plot = n_test
        x_indices = np.arange(n_data_to_plot)

        # Adjust event_index to be relative to D_test/Q_test if plotting only test data
        if event_index is not None and max(event_index) > n_test:
            # Filter event_index to only include those relevant to the test set
            # and adjust their values to be 0-indexed within the test set
            adjusted_event_index = [idx - n_train for idx in event_index if idx >= n_train]
        else:
            adjusted_event_index = event_index
        
        # Determine colors for D plot (test data only)
        if adjusted_event_index is not None:
            colors_d = ['red' if i in adjusted_event_index else 'blue' for i in range(n_data_to_plot)]
        else:
            colors_d = ['blue'] * n_data_to_plot
        
        # Add opacity to colors
        if opacity is not None:
            if len(opacity) == len(colors_d):
                colors_d = to_rgba_array(colors_d, alpha=opacity)
            elif len(opacity) == n_train + n_test:
                colors_d = to_rgba_array(colors_d, alpha=opacity[n_train:])

        # Plot D_test
        ax_d.bar(x_indices, D_to_plot, color=colors_d, label='Test')

        # Determine colors for Q plot (test data only)
        if adjusted_event_index is not None:
            colors_q = ['red' if i in adjusted_event_index else 'green' for i in range(n_data_to_plot)]
        else:
            colors_q = ['green'] * n_data_to_plot
        
        # Add opacity to colors
        if opacity is not None:
            if len(opacity) == len(colors_q):
                colors_q = to_rgba_array(colors_q, alpha=opacity)
            elif len(opacity) == n_train + n_test:
                colors_q = to_rgba_array(colors_q, alpha=opacity[n_train:])

        # Plot Q_test
        ax_q.bar(x_indices, Q_to_plot, color=colors_q, label='Test')

    # --- Common plotting elements for D ---
    ax_d.set_title("D-statistic")
    ax_d.set_ylabel('D')
    if np.isscalar(threshold_D):
        ax_d.axhline(y=threshold_D, linestyle='--', label='D threshold', color='red')
    else:
        for i, th in enumerate(threshold_D):
            label = f'D threshold $\\alpha$={alpha[i]}' if alpha is not None and i < len(alpha) else 'D threshold'
            ax_d.axhline(y=th, linestyle='--', label=label, color='red')
    if logscale:
        ax_d.set_yscale('log')
    ax_d.legend(loc='upper left')
    ax_d.grid(True)

    # --- Common plotting elements for Q ---
    ax_q.set_title('Q-statistic')
    ax_q.set_ylabel('Q')
    if np.isscalar(threshold_Q):
        ax_q.axhline(y=threshold_Q, linestyle='--', label='Q threshold', color='red')
    else:
        for i, th in enumerate(threshold_Q):
            label = f'Q threshold ({type_q}) $\\alpha$={alpha[i]}' if alpha is not None and i < len(alpha) else 'Q threshold'
            ax_q.axhline(y=th, linestyle='--', label=label, color='red')
    if logscale:
        ax_q.set_yscale('log')
    ax_q.legend(loc='upper left')
    ax_q.grid(True)

    # Apply x-axis labels if provided
    if labels is not None:
        if len(labels) == n_train + n_test and plot_train == False:
            labels = labels[n_train:]
        
        if len(labels) != n_data_to_plot:
            print(f"Warning: Length of 'labels' ({len(labels)}) does not match the number of plotted data points ({n_data_to_plot}). Labels will not be applied.")
        else:
            ax_d.set_xticks(x_indices)
            ax_d.set_xticklabels(labels, rotation=45, ha='right') # Rotate for better visibility
            ax_q.set_xticks(x_indices)
            ax_q.set_xticklabels(labels, rotation=45, ha='right')
            ax_q.tick_params(axis='x', which='major', pad=10) # Add some padding for rotated labels

    return fig, [ax_d, ax_q]


def plot_tscore(T, threshold_quantiles=None, logscale=False, event_index=None, labels=None, opacity=None, ax=None):
    """
    Plots T-score values.
    Highlights bars associated with `event_index` in red.

    :param T: Vector of T values.
    :param threshold_quantiles: Quantiles used as thresholds.
    :param logscale: If True, apply a logarithmic scale to the y-axis. Defaults to False.
    :param event_index: Optional list/array of indices to highlight in red. Defaults to None.
    :param labels: Optional list of strings for x-axis tick labels. Defaults to None.
    :param opacity: Optional list of opacity values for each bar. Defaults to None.
    :param ax: Optional matplotlib Axes object to plot on.
               If None, a new Figure and Axes will be created. Defaults to None.
    
    :return: A tuple (Figure, Axes).
    """
    if isinstance(event_index, int):
        event_index = [event_index]
    
    if threshold_quantiles is not None:
        if not isinstance(threshold_quantiles, list):
            threshold_quantiles = [threshold_quantiles]

    fig = None # Initialize fig to None, will be set if a new figure is created

    if ax is None:
        # Create a new figure and two subplots if no axes are provided
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure # Get the figure from the provided axes

    n_data = len(T)
    x_indices = np.arange(n_data)

    # --- Plotting T ---
    # Determine colors
    if event_index is not None:
        colors_d = ['red' if i in event_index else 'blue' for i in range(n_data)]
    else:
        colors_d = 'blue'

    # Add opacity to colors
    if opacity is not None and len(opacity) == len(T):
        colors_d = to_rgba_array(colors_d, alpha=opacity)
    
    ax.bar(x_indices, T, color=colors_d, label='T')

    if threshold_quantiles is not None:
        # Thresholds
        threshold_T = np.quantile(T, threshold_quantiles)
        # Handle thresholds for D
        if len(threshold_T) == 1:
            ax.axhline(y=threshold_T, linestyle='--', label='Threshold', color='red')
        else:
            for i, th in enumerate(threshold_T):
                ax.axhline(y=th, linestyle='--', label=f"Percentile {100 * threshold_quantiles[i]}", color='red')

    ax.set_title("T-score")
    ax.set_ylabel('T')
    if logscale:
        ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True)

    # Apply x-axis labels if provided
    if labels is not None:
        if len(labels) != n_data:
            print(f"Warning: Length of 'labels' ({len(labels)}) does not match the number of plotted data points ({n_data}). Labels will not be applied.")
        else:
            ax.set_xticks(x_indices)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.tick_params(axis='x', which='major', pad=10)

    return fig, ax


def plot_tscore_tt(T_train, T_test, threshold_quantiles=None, logscale=False, 
               event_index=None, labels=None, plot_train=True, opacity=None, ax=None):
    """
    Plots the T-score results for training and test datasets, supporting one or multiple thresholds.
    Highlights bars associated with `event_index` in red.

    :param T_train: T-score values for the training data.
    :param D_test: T-score values values for the test data.
    :param threshold_quantiles: Threshold(s) for Q-statistic. Can be a scalar or a list.
        :param T: Vector of T values.
    :param threshold_quantiles: Quantiles used as thresholds.
    :param logscale: If True, apply a logarithmic scale to the y-axis. Defaults to False.
    :param event_index: Optional list/array of indices to highlight in red. Defaults to None.
    :param labels: Optional list of strings for x-axis tick labels. Defaults to None.
    :param plot_train: If True, both training and test data are plotted. If False, only
                       test data is plotted. Defaults to True.
    :param opacity: Optional list of opacity values for each bar. Defaults to None.
    :param ax: Optional matplotlib Axes object to plot on.
               If None, a new Figure and Axes will be created. Defaults to None.

    :return: A tuple (Figure, Axes).
    """
    if isinstance(event_index, int):
        event_index = [event_index]
    
    if threshold_quantiles is not None:
        if not isinstance(threshold_quantiles, list):
            threshold_quantiles = [threshold_quantiles]

    fig = None # Initialize fig to None, will be set if a new figure is created

    if ax is None:
        # Create a new figure and two subplots if no axes are provided
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure # Get the figure from the provided axes

    n_train = len(T_train)
    n_test = len(T_test)

    if plot_train:
        # Concatenate train and test data for plotting
        T_to_plot = np.concatenate([T_train, T_test])
        n_data_to_plot = len(T_to_plot)
        x_indices = np.arange(n_data_to_plot)

        # Determine colors for D plot
        if event_index is not None:
            colors_d = ['red' if i in event_index else ('blue' if i < n_train else 'orange') for i in range(n_data_to_plot)]
        else:
            colors_d = ['blue' if i < n_train else 'orange' for i in range(n_data_to_plot)]
        
        # Add opacity to colors
        if opacity is not None and len(opacity) == len(colors_d):
            colors_d = to_rgba_array(colors_d, alpha=opacity)

        # Plot D_train and D_test
        ax.bar(np.arange(n_train), T_train, color=colors_d[:n_train], label='Train')
        ax.bar(np.arange(n_train, n_data_to_plot), T_test, color=colors_d[n_train:], label='Test')
        ax.axvline(x=n_train - 0.5, color='black', linestyle='--', label='Train/Test Split')

    else: # Only plot test data
        T_to_plot = T_test
        n_data_to_plot = n_test
        x_indices = np.arange(n_data_to_plot)

        # Adjust event_index to be relative to T_test if plotting only test data
        if event_index is not None and max(event_index) > n_test:
            # Filter event_index to only include those relevant to the test set
            # and adjust their values to be 0-indexed within the test set
            adjusted_event_index = [idx - n_train for idx in event_index if idx >= n_train]
        else:
            adjusted_event_index = event_index
        
        # Determine colors for plot (test data only)
        if adjusted_event_index is not None:
            colors = ['red' if i in adjusted_event_index else 'blue' for i in range(n_data_to_plot)]
        else:
            colors = ['blue'] * n_data_to_plot
        
        # Add opacity to colors
        if opacity is not None:
            if len(opacity) == len(colors):
                colors = to_rgba_array(colors, alpha=opacity)
            elif len(opacity) == n_train + n_test:
                colors = to_rgba_array(colors, alpha=opacity[n_train:])

        # Plot T_test
        ax.bar(x_indices, T_to_plot, color=colors, label='Test')

    # Thresholds
    if threshold_quantiles is not None:
        # Thresholds
        threshold_T = np.quantile(T_train, threshold_quantiles)
        # Handle thresholds
        if len(threshold_T) == 1:
            ax.axhline(y=threshold_T, linestyle='--', label='Threshold', color='red')
        else:
            for i, th in enumerate(threshold_T):
                ax.axhline(y=th, linestyle='--', label=f"Percentile {100 * threshold_quantiles[i]}", color='red')
                ax.set_title("T-score")
    ax.set_ylabel('T')
    if logscale:
        ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True)

    # Apply x-axis labels if provided
    if labels is not None:
        if len(labels) == n_train + n_test and plot_train == False:
            labels = labels[n_train:]
        
        if len(labels) != n_data_to_plot:
            print(f"Warning: Length of 'labels' ({len(labels)}) does not match the number of plotted data points ({n_data_to_plot}). Labels will not be applied.")
        else:
            ax.set_xticks(x_indices)
            ax.set_xticklabels(labels, rotation=45, ha='right') # Rotate for better visibility
            ax.tick_params(axis='x', which='major', pad=10) # Add some padding for rotated labels

    return fig, ax


def plot_U_tt(U_train, U_test, percentile=None, 
                alpha=None, logscale=False, event_index=None, 
                labels=None, plot_train=True, opacity=None, ax=None):
    """
    Plots the U-square results for
    training and test datasets, supporting one or multiple thresholds.
    Highlights bars associated with `event_index` in red.

    :param U_train: U-square for the training data.
    :param U_test: U-square for the test data.
    :param percentile: Percentile to use for the threshold calculation.
    :param alpha: Optional list/array of alpha values corresponding to thresholds, used for labels.
                  Defaults to None.
    :param logscale: If True, apply a logarithmic scale to the y-axis. Defaults to False.
    :param event_index: Optional list/array of indices to highlight in red. Defaults to None.
    :param labels: Optional list of strings for x-axis tick labels. Defaults to None.
    :param plot_train: If True, both training and test data are plotted. If False, only
                       test data is plotted. Defaults to True.
    :param opacity: List of opacity values for each bar. Defaults to None.
    :param ax: Optional list or tuple of two matplotlib Axes objects ([ax_D, ax_Q]) to plot on.
               If None, a new Figure and two Axes will be created. Defaults to None.
    :return: A tuple (Figure, list_of_Axes).
    """
    if isinstance(event_index, int):
        event_index = [event_index]
        
    fig = None # Initialize fig to None, will be set if a new figure is created

    if ax is None:
        # Create a new figure and two subplots if no axes are provided
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    else:
        fig = ax.figure # Get the figure from the provided axes

    n_train = len(U_train)
    n_test = len(U_test)

    if isinstance(percentile, (list, tuple)):
        assert len(percentile) == 2, "Provide two percentiles"
        top_percentile = percentile[0]
        bot_percentile = percentile[1]
    else: top_percentile, bot_percentile = percentile, percentile

    top_threshold_U, bot_threshold_U=None, None
    if top_percentile is not None:
        top_threshold_U = np.percentile((U_train[U_train>0]), 100*top_percentile)
    if bot_percentile is not None:
        bot_threshold_U = -np.percentile((-U_train[U_train<0]), 100*bot_percentile)

    if plot_train:
        # Concatenate train and test data for plotting
        U_to_plot = np.concatenate([U_train, U_test])
        n_data_to_plot = len(U_to_plot)
        x_indices = np.arange(n_data_to_plot)

        # Determine colors for U plot
        if event_index is not None:
            colors_d = ['red' if i in event_index else ('blue' if i < n_train else 'orange') for i in range(n_data_to_plot)]
        else:
            colors_d = ['blue' if i < n_train else 'orange' for i in range(n_data_to_plot)]
        
        # Add opacity to colors
        if opacity is not None and len(opacity) == len(colors_d):
            colors_d = to_rgba_array(colors_d, alpha=opacity)

        # Plot U_train and U_test
        ax.bar(np.arange(n_train), U_train, color=colors_d[:n_train], label='Train')
        ax.bar(np.arange(n_train, n_data_to_plot), U_test, color=colors_d[n_train:], label='Test')
        ax.axvline(x=n_train - 0.5, color='black', linestyle='--', label='Train/Test Split')

        # Determine colors for Q plot
        if event_index is not None:
            colors_q = ['red' if i in event_index else ('green' if i < n_train else 'orange') for i in range(n_data_to_plot)]
        else:
            colors_q = ['green' if i < n_train else 'orange' for i in range(n_data_to_plot)]

        # Add opacity to colors
        if opacity is not None and len(opacity) == len(colors_q):
            colors_q = to_rgba_array(colors_q, alpha=opacity)


    else: # Only plot test data
        U_to_plot = U_test
        n_data_to_plot = n_test
        x_indices = np.arange(n_data_to_plot)

        # Adjust event_index to be relative to U_test if plotting only test data
        if event_index is not None and max(event_index) > n_test:
            # Filter event_index to only include those relevant to the test set
            # and adjust their values to be 0-indexed within the test set
            adjusted_event_index = [idx - n_train for idx in event_index if idx >= n_train]
        else:
            adjusted_event_index = event_index
        
        # Determine colors for D plot (test data only)
        if adjusted_event_index is not None:
            colors_d = ['red' if i in adjusted_event_index else 'blue' for i in range(n_data_to_plot)]
        else:
            colors_d = ['blue'] * n_data_to_plot
        
        # Add opacity to colors
        if opacity is not None:
            if len(opacity) == len(colors_d):
                colors_d = to_rgba_array(colors_d, alpha=opacity)
            elif len(opacity) == n_train + n_test:
                colors_d = to_rgba_array(colors_d, alpha=opacity[n_train:])

        # Plot U_test
        ax.bar(x_indices, U_to_plot, color=colors_d, label='Test')

        # Determine colors for Q plot (test data only)
        if adjusted_event_index is not None:
            colors_q = ['red' if i in adjusted_event_index else 'green' for i in range(n_data_to_plot)]
        else:
            colors_q = ['green'] * n_data_to_plot
        
        # Add opacity to colors
        if opacity is not None:
            if len(opacity) == len(colors_q):
                colors_q = to_rgba_array(colors_q, alpha=opacity)
            elif len(opacity) == n_train + n_test:
                colors_q = to_rgba_array(colors_q, alpha=opacity[n_train:])

    # --- Common plotting elements for D ---
    ax.set_ylabel('U square')
    if np.isscalar(top_threshold_U):
        ax.axhline(y=top_threshold_U, linestyle='--', label='U threshold - Top', color='red')
    if np.isscalar(bot_threshold_U):
        ax.axhline(y=bot_threshold_U, linestyle='--', label='U threshold - Bottom', color='red')

    if logscale:
        ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True)


    # Apply x-axis labels if provided
    if labels is not None:
        if len(labels) == n_train + n_test and plot_train == False:
            labels = labels[n_train:]
        
        if len(labels) != n_data_to_plot:
            print(f"Warning: Length of 'labels' ({len(labels)}) does not match the number of plotted data points ({n_data_to_plot}). Labels will not be applied.")
        else:
            ax.set_xticks(x_indices)
            ax.set_xticklabels(labels, rotation=45, ha='right') # Rotate for better visibility
            ax.tick_params(axis='x', which='major', pad=10) # Add some padding for rotated labels


    return fig, ax