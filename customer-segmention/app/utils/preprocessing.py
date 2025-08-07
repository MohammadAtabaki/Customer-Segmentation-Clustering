import pandas as pd
from datetime import timedelta

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean transactional dataset.
    - Removes nulls
    - Keeps positive quantities
    - Converts dates
    """
    df = pd.read_excel(file_path) if file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)

    # Remove null CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Keep only positive quantities and valid prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) features.
    """
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                   # Frequency
        'TotalPrice': 'sum'                                       # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    return rfm.reset_index()
