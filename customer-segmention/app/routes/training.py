from fastapi import APIRouter, Query
from app.utils.clustering import *
from app.utils.evaluation import evaluate_clustering
from app.utils.model_io import save_model
from app.state import data_store

router = APIRouter()

@router.post("/")
def train_model(method: str = Query("kmeans", enum=["kmeans", "gmm", "hierarchical", "dbscan"])):
    if data_store["rfm_clv"] is None:
        return {"error": "No dataset available."}

    df = data_store["rfm_clv"]
    scaled, scaler = normalize_features(df, ["Recency", "Frequency", "Monetary", "CLV"])

    if method == "kmeans":
        best_k, labels, model, score = train_kmeans_auto(scaled)
    elif method == "gmm":
        best_k, labels, model, score = train_gmm_auto(scaled)
    elif method == "hierarchical":
        best_k, labels, score = train_hierarchical_auto(scaled)
        model = None
    elif method == "dbscan":
        labels, model = train_dbscan(scaled, eps=0.5, min_samples=5)
        best_k = len(set(labels)) - (1 if -1 in labels else 0)
        score = silhouette_score(scaled, labels) if best_k > 1 else 0

    data_store.update({
        "scaler": scaler,
        "model": model,
        "labels": labels,
        "best_k": best_k,
        "method": method
    })

    if method in ["kmeans", "gmm"]:
        save_model(model, scaler, df)

    return {
        "status": "trained_and_saved" if method in ["kmeans", "gmm"] else "trained",
        "method": method,
        "best_k": best_k,
        "silhouette_score": score,
        "evaluation": evaluate_clustering(scaled, labels) if best_k > 1 else {}
    }
