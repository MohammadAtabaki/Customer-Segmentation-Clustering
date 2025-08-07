from fastapi import FastAPI
from app.routes import upload, training, prediction, visualization
from app.utils.model_io import load_model
from app.state import data_store


app = FastAPI(title="Customer Segmentation API", version="2.0")



@app.on_event("startup")
def load_pretrained_model():
    """Load pre-trained model and data if available."""
    model, scaler, rfm_df = load_model()
    if model is not None:
        data_store.update({
            "model": model,
            "scaler": scaler,
            "rfm_clv": rfm_df,
            "labels": model.predict(scaler.transform(rfm_df[["Recency","Frequency","Monetary","CLV"]])),
            "best_k": len(set(model.predict(scaler.transform(rfm_df[["Recency","Frequency","Monetary","CLV"]])))),
            "method": "kmeans"
        })
        print("✅ Pre-trained model loaded.")
    else:
        print("⚠ No pre-trained model found. Waiting for upload.")
        
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])

@app.get("/")
def root():
    return {"message": "Customer Segmentation API is running"}
