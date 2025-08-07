from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(data, labels) -> dict:
    """
    Evaluate clustering using Silhouette & Davies-Bouldin scores.
    """
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin
    }
