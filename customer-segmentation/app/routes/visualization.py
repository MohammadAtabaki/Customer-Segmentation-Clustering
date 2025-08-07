from fastapi import APIRouter
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from app.state import data_store
from app.utils.clustering import *
from app.utils.evaluation import evaluate_clustering
from fastapi.responses import FileResponse
import os

router = APIRouter()

@router.get("/silhouette-curve")
def silhouette_curve():
    """
    Generate silhouette curve for the active method.
    """
    if data_store["rfm_clv"] is None:
        return {"error": "No data available."}

    method = data_store.get("method", "kmeans")
    df = data_store["rfm_clv"]
    scaled, _ = normalize_features(df, ["Recency", "Frequency", "Monetary", "CLV"])
    scores = []
    k_values = range(2, 11)

    for k in k_values:
        if method == "kmeans":
            _, labels, _, _ = train_kmeans_auto(scaled, k_range=(k, k))
        elif method == "gmm":
            _, labels, _, _ = train_gmm_auto(scaled, k_range=(k, k))
        elif method == "hierarchical":
            _, labels, _ = train_hierarchical_auto(scaled, k_range=(k, k))
        else:
            break
        scores.append(silhouette_score(scaled, labels))

    return {
        "method": method,
        "k_values": list(k_values),
        "silhouette_scores": scores
    }

@router.get("/segmentation-scatter")
def segmentation_scatter():
    if data_store["labels"] is None:
        return {"error": "No clustering results."}

    df = data_store["rfm_clv"].copy()
    df["Cluster"] = data_store["labels"]

    # If DBSCAN, handle noise (-1)
    if "Cluster" in df.columns:
        df["Cluster"] = df["Cluster"].astype(str)

    scatter = df[["Frequency", "Monetary", "Cluster"]].to_dict(orient="records")
    return {"scatter": scatter}


@router.get("/download/csv")
def download_clustered_csv():
    if data_store["rfm_clv"] is None or data_store["labels"] is None:
        return {"error": "No trained model available."}

    df = data_store["rfm_clv"].copy()
    df["Cluster"] = data_store["labels"]
    path = "app/utils/models/latest_clustered.csv"
    df.to_csv(path, index=False)
    return FileResponse(path, media_type="text/csv", filename="clustered_dataset.csv")

@router.get("/download/model")
def download_model():
    method = data_store.get("method", "kmeans")
    model_path = "app/utils/models/kmeans_model.pkl" if method == "kmeans" else "app/utils/models/gmm_model.pkl"

    if not os.path.exists(model_path):
        return {"error": f"No saved model found for method: {method}"}
    
    return FileResponse(model_path, media_type="application/octet-stream", filename=os.path.basename(model_path))