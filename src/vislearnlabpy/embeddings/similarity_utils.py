# Adapted (more so stolen) from https://github.com/ViCCo-Group/thingsvision/blob/master/thingsvision/core/rsa/helpers.py
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

Array = np.ndarray

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def squared_dists(X: Array) -> Array:
    """Compute squared l2-distances between two feature representations in parallel."""
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(X[i] - X[j]) ** 2
    return D

def gaussian_kernel(X: Array) -> Array:
    """Compute dissimilarity matrix based on the RBF kernel."""
    D = squared_dists(X)
    return np.exp(-D / np.mean(D))

def correlation_matrix(X: Array, a_min: float = -1.0, a_max: float = 1.0) -> Array:
    """Compute dissimilarity matrix based on correlation distance (on the matrix-level)."""
    F_c = X - X.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom).clip(min=a_min, max=a_max)
    return corr_mat

def cosine_matrix(X: Array, a_min: float = -1.0, a_max: float = 1.0) -> Array:
    """Compute dissimilarity matrix based on cosine distance (on the matrix-level)."""
    num = X @ X.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(X, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    cos_mat = (num / denom).clip(min=a_min, max=a_max)
    return cos_mat

def compute_rdm(X: Array, method: str) -> Array:
    """Compute representational dissimilarity matrix based on some distance measure.

    Parameters
    ----------
    X : ndarray
        Input array. Feature matrix of size n x p,
        where n corresponds to the number of observations
        and p is the feature dimensionaltiy.
    method : str
        Distance metric (e.g., correlation, cosine).

    Returns
    -------
    output : ndarray
        Returns the representational dissimilarity matrix.
    """
    methods = ["correlation", "cosine", "euclidean", "gaussian"]
    assert method in methods, f"\nMethod to compute RDM must be one of {methods}.\n"
    if method == "euclidean":
        rdm = squareform(pdist(X, method))
        return rdm
    else:
        if method == "correlation":
            rsm = correlation_matrix(X)
        elif method == "cosine":
            rsm = cosine_matrix(X)
        elif method == "gaussian":
            rsm = gaussian_kernel(X)
    return 1 - rsm

def correlate_rdms(
    rdm_1: Array,
    rdm_2: Array,
    correlation: str = "pearson",
) -> float:
    """Correlate the upper triangular parts of two distinct RDMs.

    Parameters
    ----------
    rdm_1 : ndarray
        First RDM.
    rdm_2 : ndarray
        Second RDM.
    correlation : str
        Correlation coefficient (e.g., Spearman, Pearson).

    Returns
    -------
    output : float
        Returns the correlation coefficient of the two RDMs.
    """
    triu_inds = np.triu_indices(len(rdm_1), k=1)
    corr_func = getattr(scipy.stats, "".join((correlation, "r")))
    rho = corr_func(rdm_1[triu_inds], rdm_2[triu_inds])[0]
    return rho

def plot_rdm(
    out_path: str,
    X: Array,
    method: str = "correlation",
    format: str = ".png",
    colormap: str = "cividis",
    show_plot: bool = False,
) -> None:
    """Compute and plot representational dissimilarity matrix based on some distance measure.

    Parameters
    ----------
    out_path : str
        Output directory. Directory where to store plots.
    X : ndarray
        Input array. Feature matrix of size n x m,
        where n corresponds to the number of observations
        and m is the number of latent dimensions.
    method : str
        Distance metric (e.g., correlation, cosine).
    format : str
        Image format in which to store visualized RDM.
    colormap : str
        Colormap for visualization of RDM.
    show_plot : bool
        Whether to show visualization of RDM after storing it to disk.

    Returns
    -------
    output : ndarray
        Returns the representational dissimilarity matrix.
    """
    rdm = compute_rdm(X, method)
    plt.figure(figsize=(10, 4), dpi=200)
    plt.imshow(rankdata(rdm).reshape(rdm.shape), cmap=getattr(plt.cm, colormap))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if not os.path.exists(out_path):
        print("\n...Output directory did not exist. Creating directories.\n")
        os.makedirs(out_path)
    plt.savefig(os.path.join(out_path, "".join(("rdm", format))))
    if show_plot:
        plt.show()
    plt.close()

# Helper functions to help with common use-case of getting embedding similarities for multiple modalities. 
# TODO: incorporate multimodal similarities. 
def csv_to_text_pairs(text_pair_csv):
    df = pd.read_csv(text_pair_csv)
    return list(df[['text1', 'text2']].itertuples(index=False, name=None))

def text_image_sims_from_csv(embedding_folder, text_pair_csv, output_csv="similarities.csv", similarity_type="cosine", model_type="clip"):
    return text_image_sims_from_folder(embedding_folder, output_csv, model_type, csv_to_text_pairs(text_pair_csv), similarity_type)

def text_image_sims_from_folder(embedding_folder, output_csv=None, text_pairs=None, similarity_type="cosine", model_type="clip"):
    from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
    image_embedding_store = EmbeddingStore.from_doc(str(Path(f"{embedding_folder}/image_embeddings/{model_type}_image_embeddings_doc.docs")))
    text_embedding_store = EmbeddingStore.from_doc(str(Path(f"{embedding_folder}/text_embeddings/{model_type}_text_embeddings_doc.docs")))
    return(text_embedding_store, image_embedding_store, output_csv, model_type, text_pairs, similarity_type)

def text_image_sims_from_stores(
    text_embedding_store,
    image_embedding_store,
    output_csv: Optional[str] = None,
    text_pairs: Optional[List[Tuple[str, str]]] = None,
    similarity_type: str = "cosine"
):   
    image_df = image_embedding_store.retrieve_similarities(similarity_type, output_csv, text_pairs)
    text_df = text_embedding_store.retrieve_similarities(similarity_type, output_csv, text_pairs)
    model_type = image_embedding_store.FeatureGenerator.model
    if output_csv == None:
        output_csv = f"{model_type}_{similarity_type}_similarities"
    image_df = image_df.rename(columns={f'{similarity_type}_similarity': 'image_similarity'})
    text_df = text_df.rename(columns={f'{similarity_type}_similarity': 'text_similarity'})
    full_df = image_df.merge(text_df, how='left', on=['text1', 'text2'])
    full_df.to_csv(output_csv, index=False)
