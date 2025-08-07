import pandas as pd
from app.utils.feature_engineering import add_clv_features

def test_add_clv_features():
    rfm = pd.DataFrame({
        "CustomerID": [1, 2],
        "Recency": [10, 20],
        "Frequency": [2, 3],
        "Monetary": [100.0, 150.0]
    })

    enriched = add_clv_features(rfm)

    assert "CLV" in enriched.columns
    assert "AvgOrderValue" in enriched.columns
    assert enriched["CLV"].iloc[0] > 0
