from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io
from app.utils.preprocessing import load_and_clean_data, compute_rfm
from app.utils.feature_engineering import add_clv_features
from app.state import data_store


router = APIRouter()

@router.post("/")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        ext = file.filename.split(".")[-1].lower()

        if ext == "csv":
            df = pd.read_csv(io.BytesIO(content))
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format. Upload .csv or .xlsx"}

        # Clean and compute features
        cleaned = df.dropna(subset=["CustomerID"])
        cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
        cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
        cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]

        rfm = compute_rfm(cleaned)
        rfm_clv = add_clv_features(rfm)

        data_store["rfm_clv"] = rfm_clv

        return {
            "status": "uploaded",
            "rows": len(rfm_clv),
            "columns": list(rfm_clv.columns),
            "sample": rfm_clv.head(5).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}
