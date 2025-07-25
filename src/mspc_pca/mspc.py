import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
from sklearn.decomposition import PCA

def UCL_D(N, A, alpha=0.05, phase = 1):
    """
    Calculates the UCL (upper control limit) for the D statistic (Tracy et al., 1992).
    
    :param N: Number of observations
    :param A: Number of principal components retained
    :param alpha: Significance level (default 0.05)
    :param phase: Control phase (1 or 2)
    :return: UCL_D (upper control limit)
    """
    if phase == 1:
        beta_a = A / 2
        beta_b = (N - A - 1) / 2

        beta_percentile = stats.beta.ppf(1-alpha, beta_a, beta_b)

        constant = ((N - 1) ** 2) / N

        UCL_D = constant * beta_percentile
    elif phase == 2:
        dfn = A
        dfd = N - A

        f_percentile = stats.f.ppf(1 - alpha, dfn, dfd)

        constant = (A * (N**2 - 1)) / (N * (N - A))

        UCL_D = constant * f_percentile
    else:
        raise ValueError("Parameter 'phase' must be 1 or 2.")

    return UCL_D


def UCL_Q(E, alpha=0.05, type_q='Jackson'):
    """
    Calculates the UCL (upper control limit) for the Q statistic (SPE)
    using the Box (1954) and Nomikos & MacGregor (1995) approach
    or alternatively the Jackson and Mudholkar (1979) approach.

    :param E: PCA reconstruction residuals
    :param alpha: Significance level (default 0.05)
    :param type_q: Type of approach ('Box' or 'Jackson')
    :return: UCL_Q (upper control limit)
    """
    if type_q == 'Box':
        Q = np.sum(E**2, axis=1)
        b = np.mean(Q)    
        v = np.var(Q, ddof=1)  

        g = v / (2 * b)
        h = 2 * b**2 / v

        chi2_percentile = stats.chi2.ppf(1 - alpha, df=h)

        UCL_Q = g * chi2_percentile

    elif type_q == 'Jackson':
        N = E.shape[0]
        covariance_matrix = np.dot(E.T, E) / (N - 1)
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)[::-1]

        lambdas = np.abs(eigenvalues)

        theta_1 = np.sum(lambdas)
        theta_2 = np.sum(lambdas**2)
        theta_3 = np.sum(lambdas**3)
        
        h_0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2**2)
        z_alpha = stats.norm.ppf(1-alpha)
        
        term_1 = (z_alpha * np.sqrt(2 * theta_2 * h_0**2)) / theta_1
        term_2 = 1
        term_3 = (theta_2 * h_0 * (h_0 - 1)) / (theta_1**2)
        

        UCL_Q = theta_1 * (term_1 + term_2 + term_3)**(1/h_0)

    else:
        raise ValueError("Unsupported Q type. Use 'Box' or 'Jackson'.")
    return UCL_Q


def DQ(X, n_components, preprocessing=2, alpha=0.05, type_q='Jackson', plot=True, logscale=False, percentile_threshold=False, event_index=None):
    """
    Calculates D-statistic and Q-statistic values for each observation.

    :param X: Input data (numpy array)
    :param n_components: Number of principal components to retain
    :param preprocessing: Type of preprocessing to apply:
        1: centering
        2: centering and scaling (default)
    :param alpha: Significance level (default 0.05)
    :param type_q: Type of approach for Q ('Box' or 'Jackson', default 'Jackson')
    :param plot: If True, plots the results
    :param logscale: If True, plots the statistics on a logarithmic scale
    :param percentile_threshold: If True, percentile `100 * (1-alpha)` is used as threshold
    :param event_index: List of indices to highlight in the plot
    """
    # Preprocessing
    if preprocessing == 1:
        mean = np.mean(X, axis=0)
        X_norm = X - mean
    elif preprocessing == 2:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1) # Bessel's correction, shouldn't affect results but is mathematically correct
        X_norm = (X - mean) / std
    else:
        X_norm = X

    # PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_norm)

    mu_t = np.mean(scores, axis=0)
    std_t = np.std(scores, axis=0, ddof=1)
    D = np.sum(((scores - mu_t) / std_t) ** 2, axis=1)

    X_norm_reconstructed = pca.inverse_transform(scores)
    residuals = X_norm - X_norm_reconstructed
    Q = np.sum(residuals ** 2, axis=1)

    thresholds_D = []
    thresholds_Q = []
    # Thresholds
    if percentile_threshold:
        thresholds_D.append(np.percentile(D, 100 * (1 - alpha)))
        thresholds_Q.append(np.percentile(Q, 100 * (1 - alpha)))
    elif np.isscalar(alpha):
        thresholds_D.append(UCL_D(X.shape[0], n_components, alpha, phase=1))
        thresholds_Q.append(UCL_Q(residuals, alpha, type_q))
    else:
        for a in alpha:
            th_D = UCL_D(X.shape[0], n_components, a, phase=1)
            th_Q = UCL_Q(residuals, a, type_q)
            thresholds_D.append(th_D)
            thresholds_Q.append(th_Q)

    if plot:
        plot_DQ(D, Q, thresholds_D, thresholds_Q, logscale=logscale, event_index=event_index)
    return D, Q, thresholds_D, thresholds_Q


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


def DQ_tt(X_train, X_test, n_components, preprocessing=2, type_q='Jackson', alpha=0.05, plot=False, 
           logscale=False, event_index=None, percentile_threshold=False):
    """
    Same as DQ, but for train/test.

    :param X_train: Training data (numpy array)
    :param X_test: Test data (numpy array)
    :param n_components: Number of principal components to retain
    :param preprocessing: Type of preprocessing to apply:
        1: centering
        2: centering and scaling (default)
    :param type_q: Type of approach for Q ('Box' or 'Jackson', default 'Jackson')
    :param alpha: Significance level(s) (default 0.05), can be a scalar or a list
    :param plot: If True, plots the results
    :param logscale: If True, plots the statistics on a logarithmic scale
    :param event_index: Index(es) of event(s) to mark on the plot
    :param percentile_threshold: If True, percentile `100 * (1-alpha)` of training data 
        is used as threshold
    """
    # Preprocessing
    if preprocessing == 1:
        mean_train = np.mean(X_train, axis=0)
        X_train_norm = X_train - mean_train
        X_test_norm = X_test - mean_train
    elif preprocessing == 2:
        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0, ddof=1)
        X_train_norm = (X_train - mean_train) / std_train
        X_test_norm = (X_test - mean_train) / std_train
    else:
        X_train_norm = X_train
        X_test_norm = X_test

    # PCA
    pca = PCA(n_components=n_components)
    scores_train = pca.fit_transform(X_train_norm)
    scores_test = pca.transform(X_test_norm)

    mu_t = np.mean(scores_train, axis=0)
    std_t = np.std(scores_train, axis=0, ddof=1) # Bessel's correction, shouldn't affect results but is mathematically correct
    D_train = np.sum(((scores_train - mu_t) / std_t) ** 2, axis=1)

    X_train_norm_reconstructed = pca.inverse_transform(scores_train)
    residuals_train = X_train_norm - X_train_norm_reconstructed
    Q_train = np.sum(residuals_train ** 2, axis=1)

    D_test = np.sum(((scores_test - mu_t) / std_t) ** 2, axis=1)
    X_test_norm_reconstructed = pca.inverse_transform(scores_test)
    residuals_test = X_test_norm - X_test_norm_reconstructed
    Q_test = np.sum(residuals_test ** 2, axis=1)

    # Thresholds
    if percentile_threshold:
        threshold_D = np.percentile(D_train, 100 * (1 - alpha))
        threshold_Q = np.percentile(Q_train, 100 * (1 - alpha))
    elif np.isscalar(alpha):
        threshold_D = UCL_D(X_train.shape[0], n_components, alpha, 2)
        threshold_Q = UCL_Q(residuals_train, alpha, type_q)
    else:
        thresholds_D = []
        thresholds_Q = []
        for a in alpha:
            th_D = UCL_D(X_train.shape[0], n_components, a, 2)
            th_Q = UCL_Q(residuals_train, a, type_q)
            thresholds_D.append(th_D)
            thresholds_Q.append(th_Q)
        threshold_D = thresholds_D
        threshold_Q = thresholds_Q

    if plot:
        plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, alpha=[alpha] if np.isscalar(alpha) else alpha, type_q=type_q, logscale=logscale, event_index=event_index)

    return D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q


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


if __name__ == "__main__":
    import matlab.engine

    def DQ_tt_MEDA(train, test, n_components, preprocessing = 0, alpha = 0.05, plot = False):
        eng = matlab.engine.start_matlab() 

        result = eng.mspcPca(
            matlab.double(train.tolist()),
            'PCs', matlab.double(list(range(1, n_components + 1))),
            'ObsTest', matlab.double(test.tolist()), 
            'Preprocessing', preprocessing,
            'PValueD', matlab.double(alpha),
            'PValueQ', matlab.double(alpha),
            'Plot', plot,
            nargout=6
        )   
        eng.quit()

        python_result = tuple(
            np.array(result[i]) if isinstance(result[i], matlab.double) else result[i]
            for i in range(len(result))
        )

        return python_result
        
    def DQ_MEDA(X, n_components, preprocessing = 0, alpha = 0.05, plot = False):
        eng = matlab.engine.start_matlab() 

        result = eng.mspcPca(
            matlab.double(X.tolist()),  
            'PCs', matlab.double(list(range(1, n_components + 1))),
            'Preprocessing', preprocessing,
            'PValueD', matlab.double(alpha),
            'PValueQ', matlab.double(alpha),
            'Plot', plot,
            nargout=6
        )   
        eng.quit()

        python_result = tuple(
            np.array(result[i]) if isinstance(result[i], matlab.double) else result[i]
            for i in range(len(result))
        )

        return python_result


    # ------------------------------------------------------------------------- TESTING -----------------------------------------------------------------------------------------
    def compare_tuples(tuple1, tuple2):
        """
        Compara elemento por elemento dos tuplas que contienen elementos numpy, 
        permitiendo ligeras diferencias debido a errores de cálculo.

        :param tuple1: Primera tupla.
        :param tuple2: Segunda tupla.
        :return: Lista de booleanos indicando si los elementos correspondientes son iguales dentro de la tolerancia.
        """
        if len(tuple1) != len(tuple2):
            raise ValueError("Las tuplas deben tener la misma longitud para ser comparadas.")
        
        atol, rtol = 1e-8, 1e-5
        while atol <= 1e-1 and rtol <= 1e-1:
            result_atol = np.prod([np.allclose(a, b, atol=atol) for a, b in zip(tuple1, tuple2)])
            result_rtol = np.prod([np.allclose(a, b, rtol=rtol) for a, b in zip(tuple1, tuple2)])
            if result_atol:
                print(f"Resultados con diferencia absoluta menor que {atol} y diferencia relativa mayor que {rtol*100}%")
                return result_atol
            if result_rtol:
                print(f"Resultados con diferencia relativa menor que {rtol*100}% y diferencia absoluta mayor que {atol}")
                return result_rtol
            atol *= 10
            rtol *= 10
        print("No se encontró una tolerancia que haga que el resultado sea verdadero.")
        return False


    def random_X(shape, dist='normal', seed=None):
        rng = np.random.default_rng(seed)
        if dist == 'normal':
            return rng.normal(size=shape)
        elif dist == 'uniform':
            return rng.uniform(-1, 1, size=shape)
        elif dist == 'exponential':
            return rng.exponential(scale=1.0, size=shape)
        elif dist == 'binary':
            return rng.integers(0, 2, size=shape)
        else:
            raise ValueError("Distribución no soportada")

    def test_thresholds(dist, n_samples=200, n_features=10, n_components=2, preprocessing=1, alphas=[0.05, 0.01]):
        X = random_X((n_samples, n_features), dist=dist, seed=42)
        # Python
        result_py = DQ(X, n_components, preprocessing, alpha=alphas, plot=False)
        # MATLAB
        result_matlab = DQ_MEDA(X, n_components, preprocessing, alpha=alphas, plot=False)
        # Compara solo los umbrales
        diffs_D = np.abs(np.array(result_py[-2]) - np.array(result_matlab[-2]))
        diffs_Q = np.abs(np.array(result_py[-1]) - np.array(result_matlab[-1]))
        coinciden = compare_tuples(result_py[-2:], result_matlab[-2:])
        return {
            "coinciden": coinciden,
            "D_py": result_py[-2],
            "D_matlab": result_matlab[-2],
            "Q_py": result_py[-1],
            "Q_matlab": result_matlab[-1],
            "diffs_D": diffs_D,
            "diffs_Q": diffs_Q
        }

    distribuciones = ['normal', 'uniform', 'exponential', 'binary']
    n_samples = 2000
    n_features = 10
    n_components_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    preprocessing_list = [0, 1, 2]
    alphas = [0.05, 0.01]

    resumen = []
    for dist in distribuciones:
        for n_components in n_components_list:
            for preprocessing in preprocessing_list:
                print(f"\nProbando: dist={dist}, n_components={n_components}, preprocessing={preprocessing}")
                res = test_thresholds(
                    dist,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_components=n_components,
                    preprocessing=preprocessing,
                    alphas=alphas
                )
                resumen.append({
                    "distribucion": dist,
                    "n_components": n_components,
                    "preprocessing": preprocessing,
                    "coinciden": res["coinciden"],
                    "D_py": res["D_py"],
                    "D_matlab": res["D_matlab"],
                    "Q_py": res["Q_py"],
                    "Q_matlab": res["Q_matlab"],
                    "diffs_D": res["diffs_D"],
                    "diffs_Q": res["diffs_Q"]
                })

    print("\nResumen de coincidencias y valores (Python vs MATLAB):")
    for r in resumen:
        estado = "OK" if r["coinciden"] else "NO"
        print(f"Distribución: {r['distribucion']}, n_components: {r['n_components']}, preprocessing: {r['preprocessing']} --> {estado}")
        print(f"  D Python:  {r['D_py']}")
        print(f"  D MATLAB:  {r['D_matlab']}")
        print(f"  Q Python:  {r['Q_py']}")
        print(f"  Q MATLAB:  {r['Q_matlab']}")
        print(f"  Diferencias D: {r['diffs_D']}")
        print(f"  Diferencias Q: {r['diffs_Q']}")

"""
Las diferencias mayores entre Python y MATLAB aparecen principalmente en el cálculo de Q (Error de Predicción) para las distribuciones **exponential** y **binary** cuando **preprocessing=0** (sin centrado/normalización).
En cambio, para preprocessing=1 o 2 (centrado o escalado), las diferencias son prácticamente nulas (del orden de 1e-15).
"""