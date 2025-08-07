import pandas as pd

def add_clv_features(rfm_df: pd.DataFrame, margin: float = 0.1, horizon: int = 12) -> pd.DataFrame:
    """
    Add Customer Lifetime Value (CLV) and derived features:
    - Avg Order Value (Monetary/Frequency)
    - CLV (avg_order_value × frequency × margin × horizon)
    """
    df = rfm_df.copy()

    df['AvgOrderValue'] = df['Monetary'] / df['Frequency']
    df['CLV'] = df['AvgOrderValue'] * df['Frequency'] * margin * horizon

    # Fill any missing values (e.g., division errors)
    df = df.fillna(0)
    return df
