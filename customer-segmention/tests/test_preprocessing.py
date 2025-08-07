import pandas as pd
from app.utils.preprocessing import compute_rfm

def test_compute_rfm():
    data = pd.DataFrame({
        "CustomerID": [1, 1, 2],
        "InvoiceDate": pd.to_datetime(["2023-08-01", "2023-08-05", "2023-08-04"]),
        "InvoiceNo": ["A001", "A002", "A003"],
        "Quantity": [2, 3, 1],
        "UnitPrice": [10.0, 15.0, 20.0],
        "TotalPrice": [20.0, 45.0, 20.0]
    })

    rfm = compute_rfm(data)

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert len(rfm) == 2
