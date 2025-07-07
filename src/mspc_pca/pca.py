import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from .ckf import ckf


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

    
def biplot(data_fitted, pc1, pc2, pca_fitted, score_labels=None, loading_labels=None):
    """
    Combines score and loading plots into a single superimposed graph,
    scaling both scores and loadings to maintain their relative positions.
    
    :param data_fitted: PCA fitted data
    :param pca_fitted: Fitted PCA model
    :param score_labels: List of labels for the score points
    :param loading_labels: List of labels for the loading points
    """
    pc1, pc2 = pc1-1, pc2-1 # Adapt human languaje to python indexes
    explained_variance = pca_fitted.explained_variance_ratio_ * 100  
    plt.figure(figsize=(8, 6))

    # Scale scores and loadings to the same scale
    scores_scaled = data_fitted / np.max(np.abs(data_fitted), axis=0)
    loadings_scaled = pca_fitted.components_ / np.max(np.abs(pca_fitted.components_), axis=1)[:, np.newaxis]

    # Plot scores
    plt.scatter(scores_scaled[:, pc1], scores_scaled[:, pc2], alpha=0.5, label='Scores', color='blue')
    if score_labels is not None:
        for i, label in enumerate(score_labels):
            plt.text(scores_scaled[i, pc1], scores_scaled[i, pc2], label, fontsize=8, color='black')

    # Plot loadings
    plt.scatter(loadings_scaled[pc1], loadings_scaled[pc2], alpha=0.5, label='Loadings', color='red')
    if loading_labels is not None:
        for i, label in enumerate(loading_labels):
            plt.text(loadings_scaled[0, i], loadings_scaled[1, i], label, fontsize=8, color='black')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

    plt.xlabel(f'PC1 ({explained_variance[pc1]:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[pc2]:.2f}%)')
    # plt.title('2D Scores and Loadings (Superimposed and Scaled)')
    plt.legend()
    plt.show()