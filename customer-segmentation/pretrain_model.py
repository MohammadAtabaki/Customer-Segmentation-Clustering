import pandas as pd
from app.utils.preprocessing import load_and_clean_data, compute_rfm
from app.utils.feature_engineering import add_clv_features
from app.utils.clustering import normalize_features, train_kmeans_auto
from app.utils.model_io import save_model

# 1. Load dataset
df = load_and_clean_data("data/online-retail-dataset.csv")

# 2. Compute RFM + CLV
rfm = compute_rfm(df)
rfm_clv = add_clv_features(rfm)

# 3. Train auto-kmeans
scaled, scaler = normalize_features(rfm_clv, ["Recency", "Frequency", "Monetary", "CLV"])
best_k, labels, model, score = train_kmeans_auto(scaled)

print(f"✅ Pre-trained model ready: k={best_k}, silhouette={score}")

# 4. Save model, scaler, and RFM dataset
save_model(model, scaler, rfm_clv)
print("💾 Model saved to app/utils/models/")
