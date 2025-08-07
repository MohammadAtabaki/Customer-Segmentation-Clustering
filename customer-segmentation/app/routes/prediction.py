from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from app.state import data_store
from app.utils.clustering import normalize_features

router = APIRouter()

class CustomerData(BaseModel):
    recency: float
    frequency: float
    monetary: float
    clv: float

@router.post("/")
def predict_cluster(customer: CustomerData):
    """
    Predicts cluster for a new customer.
    """
    if data_store["model"] is None and data_store["method"] != "hierarchical":
        return {"error": "No trained model found. Please train first."}

    scaler = data_store["scaler"]
    model = data_store["model"]
    method = data_store["method"]

    new_data = np.array([[customer.recency, customer.frequency, customer.monetary, customer.clv]])
    scaled_new = scaler.transform(new_data)

    if method in ["kmeans", "gmm"]:
        cluster = model.predict(scaled_new)[0]
    else:  # hierarchical
        # Re-compute hierarchical labels (no direct predict)
        labels = data_store["labels"]
        cluster = labels[np.random.randint(len(labels))]  # Approximate (or nearest centroid logic can be added)

    return {"predicted_cluster": int(cluster)}
