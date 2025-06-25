import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def UCL_D(N, A, alpha=0.05, phase = 1):
    """
    Calcula el UCL (umbral superior de control) para el estadístico D (Tracy et al., 1992).
    
    :param N: Número de observaciones
    :param A: Número de componentes principales retenidos
    :param alpha: Nivel de significancia (default 0.05)
    :param phase: Fase del control (1 o 2)
    :return: UCL_D (umbral superior de control)
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
        raise ValueError("Parámetro 'phase' debe ser 1 o 2.")

    return UCL_D

def UCL_Q(E, alpha=0.05, type_q='Jackson'):
    """
    Calcula el UCL (umbral superior de control) para el estadístico Q (SPE)
    usando la aproximación de Box (1954) y Nomikos & MacGregor (1995)
    ó alternativamente la aproximación de Jackson y Mudholkar (1979).
    :param E: Residuos de la reconstrucción PCA
    :param alpha: Nivel de significancia (default 0.05)
    :param type_q: Tipo de aproximación ('Box' o 'Jackson')
    :return: UCL_Q (umbral superior de control)
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
        raise ValueError("Tipo de Q no soportado. Usa 'Box' o 'Jackson'.")
    return UCL_Q

def DyQ(X, n_components, preprocessing=2, alpha=0.05, type_q='Jackson', plot=True, logscale=False, percentile_threshold=False, event_index=None):
    """
    Calcula la distancia de Hotelling (D) y el error de predicción (Q) para cada observación.
    :param X: Datos de entrada (numpy array)
    :param n_components: Número de componentes principales a retener
    :param preprocessing: Tipo de preprocesado a aplicar:
        0: sin preprocesado
        1: centrado
        2: centrado y escalado (default)
    :param alpha: Nivel de significancia (default 0.05)
    :param type_q: Tipo de aproximación para Q ('Box' o 'Jackson', default 'Jackson')
    :param plot: Si True, grafica los resultados
    :param logscale: Si True, grafica los estadísticos en escala logarítmica
    """
    # Preprocesado
    if preprocessing == 1:
        mean = np.mean(X, axis=0)
        X_norm = X - mean
    elif preprocessing == 2:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1) # Corrección de Bessel, no debería afectar a los resultados pero es matemáticamente correcto
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
    # Umbrales
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
        plot_DyQ(D, Q, thresholds_D, thresholds_Q, logscale=logscale, event_index=event_index)
    return D, Q, thresholds_D, thresholds_Q

def plot_DyQ(D, Q, threshold_D, threshold_Q, logscale=False, event_index=None):
    """
    :param D: vector de distancias de Hotelling
    :param Q: vector de errores de predicción
    :param threshold_D: umbral(es) para D puede ser escalar o lista
    :param threshold_Q: umbral(es) para Q puede ser escalar o lista
    :param logscale: Si True, grafica en escala logarítmica los estadísticos
    :param event_index: Índice(s) de evento(s) a marcar en la gráfica (opcional)
    Permite graficar uno o varios umbrales (threshold_D y threshold_Q ).
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    if event_index is not None:
        colors = ['red' if i in event_index else 'blue' for i in range(len(D))]
    else:
        colors = 'blue'
    plt.bar(range(len(D)), D, color=colors, alpha=1, label='D (Distancia de Hotelling)')
    # Soporta varios umbrales
    if np.isscalar(threshold_D):
        plt.axhline(y=threshold_D, linestyle='--', label='Umbral D', color='red')
    else:
        for th, a in zip(threshold_D, np.atleast_1d(getattr(threshold_D, 'alpha', [None]*len(threshold_D)))):
            label = f'Umbral D α={a}' if a is not None else 'Umbral D'
            plt.axhline(y=th, linestyle='--', label=label, color='red')
    plt.title('Distancia de Hotelling (D)')
    plt.ylabel('D')
    if logscale:
        plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(2, 1, 2)
    if event_index is not None:
        colors = ['red' if i in event_index else 'green' for i in range(len(Q))]
    else:
        colors = 'green'
    plt.bar(range(len(Q)), Q, color=colors, alpha=1, label='Q (Error de Predicción)')
    if np.isscalar(threshold_Q):
        plt.axhline(y=threshold_Q, linestyle='--', label='Umbral Q', color='red')
    else:
        for th, a in zip(threshold_Q, np.atleast_1d(getattr(threshold_Q, 'alpha', [None]*len(threshold_Q)))):
            label = f'Umbral Q α={a}' if a is not None else 'Umbral Q'
            plt.axhline(y=th, linestyle='--', label=label, color='red')
    plt.title('Error de Predicción (Q)')
    plt.ylabel('Q')
    if logscale:
        plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

def DyQ_tt(X_train, X_test, n_components, preprocessing=2, type_q='Jackson', alpha=0.05, plot=False, logscale=False, event_index=None, percentile_threshold=False):
    """
    Igual que DyQ, pero para train/test.
    :param X_train: Datos de entrenamiento (numpy array)
    :param X_test: Datos de prueba (numpy array)
    :param n_components: Número de componentes principales a retener
    :param preprocessing: Tipo de preprocesado a aplicar:
        0: sin preprocesado
        1: centrado
        2: centrado y escalado (default)
    :param type_q: Tipo de aproximación para Q ('Box' o 'Jackson', default 'Jackson')
    :param alpha: Nivel(es) de significancia (default 0.05), puede ser un escalar o una lista
    :param plot: Si True, grafica los resultados
    :param logscale: Si True, grafica los estadísticos en escala logarítmica
    :param event_index: Índice(s) de evento(s) a marcar en la gráfica
    :param percentile_threshold: Si True, calcula los umbrales por percentil a partir de los datos de prueba
    """
    # Preprocesado
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
    std_t = np.std(scores_train, axis=0, ddof=1) # Corrección de Bessel, no debería afectar a los resultados pero es matemáticamente correcto
    D_train = np.sum(((scores_train - mu_t) / std_t) ** 2, axis=1)

    X_train_norm_reconstructed = pca.inverse_transform(scores_train)
    residuals_train = X_train_norm - X_train_norm_reconstructed
    Q_train = np.sum(residuals_train ** 2, axis=1)

    D_test = np.sum(((scores_test - mu_t) / std_t) ** 2, axis=1)
    X_test_norm_reconstructed = pca.inverse_transform(scores_test)
    residuals_test = X_test_norm - X_test_norm_reconstructed
    Q_test = np.sum(residuals_test ** 2, axis=1)

    # Umbrales
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
        plot_DyQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, alpha=[alpha] if np.isscalar(alpha) else alpha, type_q=type_q, logscale=logscale, event_index=event_index)

    return D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q

def plot_DyQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, alpha=None, type_q='Jackson', logscale=False, event_index=None):
    """
    Grafica los resultados de D y Q para train y test, soportando uno o varios umbrales.
    Marca en rojo las barras asociadas a event_index.
    """
    D_all = np.concatenate([D_train, D_test])
    Q_all = np.concatenate([Q_train, Q_test])
    n_train = len(D_train)

    plt.figure(figsize=(12, 6))

    # Gráfico de D
    plt.subplot(2, 1, 1)
    if event_index is not None:
        colors = ['red' if i in event_index else ('blue' if i < n_train else 'yellow') for i in range(len(D_all))]
    else:
        colors = ['blue' if i < n_train else 'yellow' for i in range(len(D_all))]
    
    plt.bar(range(n_train), D_train, color=colors[:n_train], alpha=1, label='Train')
    plt.bar(range(n_train, len(D_all)), D_test, color=colors[n_train:], alpha=1, label='Test')
    if np.isscalar(threshold_D):
        plt.axhline(y=threshold_D, linestyle='--', label='Umbral D', color='red')
    else:
        for i, th in enumerate(threshold_D):
            label = f'Umbral D α={alpha[i]}' if alpha is not None else 'Umbral D'
            plt.axhline(y=th, linestyle='--', label=label, color='red')
    plt.title('Distancia de Hotelling (D)')
    plt.ylabel('D')
    plt.axvline(x=n_train-0.5, color='black', linestyle='--', label='Separador Train/Test')
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.grid()

    # Gráfico de Q
    plt.subplot(2, 1, 2)
    if event_index is not None:
        colors = ['red' if event_index and i in event_index else ('green' if i < n_train else 'yellow') for i in range(len(D_all))]
    else:
        colors = ['green' if i < n_train else 'yellow' for i in range(len(D_all))]

    plt.bar(range(n_train), Q_train, color=colors[:n_train], alpha=1, label='Train')
    plt.bar(range(n_train, len(Q_all)), Q_test, color=colors[n_train:], alpha=1, label='Test')
    if np.isscalar(threshold_Q):
        plt.axhline(y=threshold_Q, linestyle='--', label='Umbral Q', color='red')
    else:
        for i, th in enumerate(threshold_Q):
            label = f'Umbral Q ({type_q}) α={alpha[i]}' if alpha is not None else 'Umbral Q'
            plt.axhline(y=th, linestyle='--', label=label, color='red')
    plt.title('Error de Predicción (Q)')
    plt.ylabel('Q')
    plt.axvline(x=n_train-0.5, color='black', linestyle='--', label='Separador Train/Test')
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import matlab.engine

    def DyQ_tt_MEDA(train, test, n_components, preprocessing = 0, alpha = 0.05, plot = False):
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
        
    def DyQ_MEDA(X, n_components, preprocessing = 0, alpha = 0.05, plot = False):
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
        result_py = DyQ(X, n_components, preprocessing, alpha=alphas, plot=False)
        # MATLAB
        result_matlab = DyQ_MEDA(X, n_components, preprocessing, alpha=alphas, plot=False)
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