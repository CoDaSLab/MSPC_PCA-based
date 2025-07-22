import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def omeda(testcs, dummy, R, OutSubspace=None, plot=True, var_labels=None, var_classes=None,
          title='oMEDA', ax=None):
    """
    Observation-based Missing data methods for Exploratory Data Analysis (oMEDA).
    The original paper is Journal of Chemometrics, 2011, 25 (11): 592-600.
    This algorithm follows the direct computation for Known Data Regression (KDR)
    missing data imputation.

    Parameters
    ----------
    testcs : numpy.ndarray
        [NxM] preprocessed bilinear data set with the observations to be compared.
    dummy : numpy.ndarray
        [Nx1] dummy variable containing weights for the observations to compare,
        and 0 for the rest of observations.
    R : numpy.ndarray
        [MxA] Matrix to perform the projection from the original to the latent subspace.
        For PCA (testcs = T * P'), this is the matrix of loadings P. For PLS
        (Y = testcs * W * inv(P' * W) * Q), this matrix is W * inv(P' * W). For the original space
        (default) the identity matrix is used.
    OutSubspace : numpy.ndarray, optional
        [MxA] Matrix to perform the projection from the latent subspace to the
        original space. For PCA (testcs = T * P'), this is the matrix of loadings P.
        For PLS (Y = testcs * W * inv(P' * W) * Q), this matrix is also P. For the
        original space the identity matrix is used. If None, Q=R is used by default.
    plot : bool, optional
        If True, plots the oMEDA vector. Default is True.
    var_labels : list of str, optional
        Labels for variables in the X-axis.
    var_classes : list or array, optional
        Class/group for each variable (used to color bars).
    title : str, optional
        Title for the plot. Default is 'oMEDA'.
    ax: Axis, optional
        Axis where the oMEDA vector will be plotted. If None, a new figure and
        axis will be created (default: None).

    Returns
    -------
    omeda_vec : numpy.ndarray
        [Mx1] oMEDA vector.
    fig : matplotlib.figure.Figure
        If plot=True, the matplotlib Figure object if a new plot was created, otherwise None.
    ax : matplotlib.axes.Axes
        If plot=True, the matplotlib Axes object where the plot was drawn.
    """

    # Set default values for Q
    if OutSubspace is None:
        Q = R
    else:
        Q = OutSubspace

    # Convert row arrays to column arrays if necessary
    if dummy.ndim == 1:
        dummy = dummy.reshape(-1, 1)
    elif dummy.shape[1] == 1 and dummy.shape[0] == 1: # Handle scalar case
        dummy = dummy.reshape(-1, 1)


    # Validate dimensions of input data
    N, M = testcs.shape

    if dummy.shape != (N, 1):
        raise ValueError(f"Dimension Error: parameter 'dummy' must be ({N}, 1). Current shape: {dummy.shape}")

    M_R, A_R = R.shape
    if M_R != M:
        raise ValueError(f"Dimension Error: parameter 'R' must have M rows ({M}). Current rows: {M_R}")

    M_Q, A_Q = Q.shape
    if M_Q != M or A_Q != A_R:
        raise ValueError(f"Dimension Error: parameter 'OutSubspace' must be ({M}, {A_R}). Current shape: {Q.shape}")

    # Main code
    ind_pos = np.where(dummy > 0)[0]
    if len(ind_pos) > 0 and np.max(dummy[ind_pos]) != 0:
        dummy[ind_pos] = dummy[ind_pos] / np.max(dummy[ind_pos])

    ind_neg = np.where(dummy < 0)[0]
    if len(ind_neg) > 0 and np.min(dummy[ind_neg]) != 0:
        dummy[ind_neg] = -dummy[ind_neg] / np.min(dummy[ind_neg])

    xA = testcs @ R @ Q.T
    sumA = xA.T @ dummy
    sum_val = testcs.T @ dummy

    omeda_vec = ((2 * sum_val - sumA) * np.abs(sumA)) / np.sqrt(dummy.T @ dummy)
    omeda_vec = omeda_vec.ravel()

    # Plot oMEDA vector
    fig = None
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure # Get the figure associated with the provided ax

        x = np.arange(len(omeda_vec))

        # Handle coloring
        if var_classes is not None:
            unique_classes = np.unique(var_classes)
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(unique_classes))]
            class_color_map = dict(zip(unique_classes, colors))
            bar_colors = [class_color_map[cls] for cls in var_classes]

            # Create legend
            patches = [mpatches.Patch(color=color, label=str(cls)) for cls, color in class_color_map.items()]
            ax.legend(handles=patches, prop={'size': 10})
        else:
            bar_colors = 'tab:blue'

        ax.bar(x, omeda_vec, color=bar_colors)

        ax.axhline(0, color='black', linewidth=1.5)  # X-axis

        # Handle variable labels
        if var_labels is not None:
            if len(var_labels) != len(omeda_vec):
                raise ValueError(f"var_labels length ({len(var_labels)}) must match number of variables ({len(omeda_vec)})")
            ax.set_xticks(x)
            label_lengths = [len(str(label)) for label in var_labels]
            rot = 90 if max(label_lengths) > 3 else 0  # rotate if labels are too long
            ax.set_xticklabels(var_labels, rotation=rot)
            ax.tick_params(axis='x', labelsize=10)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Variables', fontsize=12)
        ax.set_ylabel('$d^2_A$', fontsize=12)
        ax.grid(True)

        return omeda_vec, fig, ax

    else:
        return omeda_vec