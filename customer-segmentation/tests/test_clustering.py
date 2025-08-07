import pandas as pd
from app.utils.clustering import normalize_features, train_kmeans_auto

def test_train_kmeans_auto():
    df = pd.DataFrame({
        "Recency": [1, 10, 5, 20, 15],
        "Frequency": [5, 2, 3, 1, 1],
        "Monetary": [100, 50, 60, 30, 40],
        "CLV": [500, 300, 350, 200, 250]
    })

    scaled, scaler = normalize_features(df, ["Recency", "Frequency", "Monetary", "CLV"])
    best_k, labels, model, score = train_kmeans_auto(scaled, k_range=(2, 4))

    assert best_k >= 2
    assert len(labels) == len(df)
    assert score > 0
