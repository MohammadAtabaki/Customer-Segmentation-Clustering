import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN

def normalize_features(df, features):
    """Scale numeric features for clustering."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler

# ------------------------------
# AUTO K-MEANS
# ------------------------------
def train_kmeans_auto(data: np.array, k_range=(2, 10)):
    """
    Train K-Means clustering with automatic k selection (Silhouette score).
    Returns: best_k, labels, model, silhouette_score
    """
    n_samples = len(data)

    # ✅ Safeguard: Prevent invalid k (must be < n_samples)
    max_k = min(k_range[1], n_samples - 1)
    min_k = max(k_range[0], 2)

    if max_k < min_k:
        raise ValueError(f"Not enough samples ({n_samples}) for clustering in the range k={k_range}")

    best_k, best_score, best_model, best_labels = None, -1, None, None

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, model, labels

    return best_k, best_labels, best_model, best_score


# ------------------------------
# AUTO GAUSSIAN MIXTURE (GMM)
# ------------------------------
def train_gmm_auto(data: np.array, k_range=(2, 10)):
    """
    Train Gaussian Mixture Model with automatic k selection (Silhouette score).
    Returns: best_k, labels, model, silhouette_score
    """
    best_k, best_score, best_model, best_labels = None, -1, None, None
    for k in range(k_range[0], k_range[1] + 1):
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, model, labels

    return best_k, best_labels, best_model, best_score

# ------------------------------
# HIERARCHICAL CLUSTERING (AUTO K)
# ------------------------------
def train_hierarchical_auto(data: np.array, k_range=(2, 10)):
    """
    Hierarchical clustering with automatic k selection using silhouette.
    """
    best_k, best_score, best_labels = None, -1, None
    for k in range(k_range[0], k_range[1] + 1):
        linkage_matrix = linkage(data, method='ward')
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(data, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    return best_k, best_labels, best_score


def train_dbscan(data, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels, model