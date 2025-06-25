import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from .ckf import ckf


def pca_resvar(data, max_components, with_std=True, with_ckf=True, exclude_zero=False):
    """
    Realiza PCA y grafica la varianza residual y la curva CKF para seleccionar el número óptimo de componentes a partir del método del codo.
    
    @param data: Matriz de datos (ventanas x frecuencias)
    @param max_components: Número de componentes principales a considerar
    @param with_std: Si True, normaliza los datos antes de PCA
    @param with_ckf: Si True, calcula la curva CKF
    @param exclude_zero: Si True, excluye los valores de varianza asociados a 0 componentes, esto es útil para visualización, pero puede afectar a la detección del "codo"
    """
    scaler = StandardScaler(with_std=with_std)
    data_norm = scaler.fit_transform(data)

    # --------------------
    # VARIANZA RESIDUAL
    # --------------------
    pca = PCA(n_components=max_components)
    data_fitted = pca.fit_transform(data_norm)
    varianza_explicada_acumulada = np.cumsum(pca.explained_variance_ratio_)
    if not exclude_zero:
        varianza_explicada_acumulada = np.insert(varianza_explicada_acumulada, 0, 0)
    varianza_residual = 1 - varianza_explicada_acumulada

    # --------------------
    # CKF
    # --------------------
    if with_ckf:
        scores = data_fitted
        loadings = pca.components_[:max_components, :]
        ckf_cumpress = ckf(data_norm, scores, loadings.T, plot=False)
        if not exclude_zero:
            ckf_cumpress = ckf_cumpress / ckf_cumpress[0]  # Normalizar CKF
        else:
            ckf_cumpress = ckf_cumpress[1:] / ckf_cumpress[0]

    # --------------------
    # GRÁFICA FINAL
    # --------------------
    plt.figure(figsize=(10, 6))
    plot_range = range(1, max_components + 1) if exclude_zero else range(0, max_components + 1)
    plt.plot(plot_range, varianza_residual, label='Varianza Residual', color='red', marker='o')
    plt.xlabel('Número de Componentes Principales', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Varianza Residual', fontsize=20)
    plt.title('Curva Varianza Residual')

    if with_ckf:
        plt.title('Curva Varianza Residual y Curva CKF')
        plt.plot(plot_range, ckf_cumpress, label='Curva CKF', color='blue', marker='s')

    plt.legend()
    plt.show()


def adjust_PCA(data, n_components, with_std=False):
    """
    Ajusta PCA con el número de componentes especificado.

    :param data: Matriz de datos (ventanas x frecuencias)
    :param n_components: Número de componentes principales
    :param with_std: Si True, normaliza los datos con desviación estándar (se realiza centrado por defecto)
    :return: Datos ajustados al modelo PCA segun parametros, PCA ajustado
    """
    scaler = StandardScaler(with_std=with_std)
    data_norm = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    data_fit = pca.fit_transform(data_norm)

    return data_fit, pca


def plot_scores_2d(data_fitted, pc1:int, pc2:int, pca_fitted):
    """
    Grafica los scores en 2D con la varianza explicada por cada componente en los ejes.
    
    :param data_fitted: Datos ajustados por PCA
    :param pca_fitted: PCA ajustado
    """
    pc1-=1
    pc2-=1
    explained_variance = pca_fitted.explained_variance_ratio_ * 100  # Convertir a porcentaje
    plt.figure(figsize=(8, 6))
    plt.scatter(data_fitted[:, pc1], data_fitted[:, pc2], alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Eje horizontal
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)  # Eje vertical
    plt.xlabel(f'PC{pc1+1} ({explained_variance[pc1]:.2f}%)', fontsize=20)
    plt.ylabel(f'PC{pc2+1} ({explained_variance[pc2]:.2f}%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Scores 2D', fontsize = 16)
    plt.show()

def plot_loadings_2d(pc1, pc2, pca_fitted):
    """
    Grafica los loadings en 2D.
    
    :param data_fitted: Datos ajustados por PCA
    :param pca_fitted: PCA ajustado
    """
    pc1-=1
    pc2-=1
    explained_variance = pca_fitted.explained_variance_ratio_ * 100  # Convertir a porcentaje
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_fitted.components_[pc1], pca_fitted.components_[pc2], alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Eje horizontal
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)  # Eje vertical
    plt.xlabel(f'PC{pc1+1} ({explained_variance[pc1]:.2f}%)', fontsize=20)
    plt.ylabel(f'PC{pc2+1} ({explained_variance[pc2]:.2f}%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Loadings 2D', fontsize = 16)
    plt.show()

def plot_scores_1d(data_fitted, pc1, pca_fitted):
    """
    Grafica los scores en 1D para una sola componente principal.
    
    :param data_fitted: Datos ajustados por PCA
    :param pc1: Índice de la componente principal (normalmente 0 para la primera)
    :param pca_fitted: PCA ajustado
    """
    pc1-=1
    explained_variance = pca_fitted.explained_variance_ratio_[pc1] * 100  # Convertir a porcentaje
    plt.figure(figsize=(8, 4))
    plt.plot(data_fitted[:, pc1], marker='o', linestyle='-', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Línea horizontal en 0
    plt.xlabel('Índice de observación')
    plt.ylabel(f'PC{pc1} ({explained_variance:.2f}%)')
    plt.title('Scores en 1D')
    plt.grid(True)
    plt.show()

def plot_loadings_1d(pc1, pca_fitted):
    """
    Grafica los loadings en 1D para una sola componente principal.
    
    :param pc1: Índice de la componente principal (normalmente 0 para la primera)
    :param pca_fitted: PCA ajustado
    """
    pc1-=1
    explained_variance = pca_fitted.explained_variance_ratio_[pc1] * 100  # Convertir a porcentaje
    plt.figure(figsize=(8, 4))
    plt.plot(pca_fitted.components_[pc1], marker='o', linestyle='-', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Línea horizontal en 0
    plt.xlabel('Índice de variable')
    plt.ylabel(f'PC{pc1+1} ({explained_variance:.2f}%)')
    plt.title('Loadings en 1D')
    plt.grid(True)
    plt.show()

    
def combine_plots(data_fitted, pc1, pc2, pca_fitted, score_labels=None, loading_labels=None):
    """
    Combina las gráficas de scores y loadings en un solo gráfico superpuesto,
    escalando tanto scores como loadings para mantener sus posiciones relativas.
    
    :param data_fitted: Datos ajustados por PCA
    :param pca_fitted: PCA ajustado
    :param score_labels: Lista de etiquetas para los puntos de scores
    :param loading_labels: Lista de etiquetas para los puntos de loadings
    """
    explained_variance = pca_fitted.explained_variance_ratio_ * 100  
    plt.figure(figsize=(8, 6))

    # Escalar scores y loadings
    scores_scaled = data_fitted / np.max(np.abs(data_fitted), axis=0)
    loadings_scaled = pca_fitted.components_ / np.max(np.abs(pca_fitted.components_), axis=1)[:, np.newaxis]

    # Gráfica de scores
    plt.scatter(scores_scaled[:, pc1], scores_scaled[:, pc2], alpha=0.5, label='Scores', color='blue')
    if score_labels is not None:
        for i, label in enumerate(score_labels):
            plt.text(scores_scaled[i, pc1], scores_scaled[i, pc2], label, fontsize=8, color='black')

    # Gráfica de loadings
    plt.scatter(loadings_scaled[pc1], loadings_scaled[pc2], alpha=0.5, label='Loadings', color='red')
    if loading_labels is not None:
        for i, label in enumerate(loading_labels):
            plt.text(loadings_scaled[0, i], loadings_scaled[1, i], label, fontsize=8, color='black')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

    plt.xlabel(f'PC1 ({explained_variance[pc1]:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[pc2]:.2f}%)')
    plt.title('Scores y Loadings en 2D (Superpuestos y Escalados)')
    plt.legend()
    plt.show()

