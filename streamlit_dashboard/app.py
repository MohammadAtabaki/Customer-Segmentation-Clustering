import streamlit as st
import pandas as pd
import plotly.express as px
from api_client import *

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation Dashboard (RFM + CLV)")

# Upload or Use Pretrained
st.header("1. Upload Dataset or Use Pretrained Model")
file = st.file_uploader("Upload CSV/Excel (optional):", type=["csv", "xlsx"])
if st.button("Load Data"):
    resp = upload_file(file)
    if "error" in resp:
        st.error(resp["error"])
    else:
        mode = resp.get("status", "unknown")
        st.success(f"✅ Loaded data ({mode}) - {resp['rows']} rows")
        st.dataframe(pd.DataFrame(resp["sample"]))

# Training with method selection
st.header("2. Train Model")
method = st.selectbox("Select clustering method", ["kmeans", "gmm", "hierarchical", "dbscan"])

if st.button("🚀 Train using selected method"):
    train_resp = train_model(method)
    if "error" in train_resp:
        st.error(train_resp["error"])
    else:
        st.success(f"Model trained! Method = {train_resp['method'].upper()} | Best k = {train_resp['best_k']}")
        st.json(train_resp)

        # Reload updated visualizations
        scatter_data = get_scatter_plot()
        silhouette_data = get_silhouette_curve()

# Download links
st.subheader("📥 Downloads")

csv_url = get_csv_download_url()
st.markdown(f"[⬇️ Download Clustered CSV]({csv_url})", unsafe_allow_html=True)

if method in ["kmeans", "gmm"]:  # Only for methods that produce model files
    model_url = get_model_download_url()
    st.markdown(f"[⬇️ Download Trained Model]({model_url})", unsafe_allow_html=True)
else:
    st.info("Trained model is not downloadable for this method.")



# Visualizations
st.header("3. Visualizations (Using Active Model)")
cols = st.columns(4)
hist = get_histograms()
if "histograms" in hist:
    for i, metric in enumerate(["Recency", "Frequency", "Monetary", "CLV"]):
        data = hist["histograms"][metric]
        df_hist = pd.DataFrame({"bin": data["bins"][:-1], "count": data["counts"]})
        with cols[i]:
            fig = px.bar(df_hist, x="bin", y="count", title=metric)
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Silhouette Curve")
sil = get_silhouette_curve()
if "silhouette_scores" in sil:
    fig = px.line(x=sil["k_values"], y=sil["silhouette_scores"], labels={"x":"k","y":"Score"})
    st.plotly_chart(fig)

st.subheader("Segmentation Scatter Plot")
scatter = get_scatter_plot()
if "scatter" in scatter:
    df_scat = pd.DataFrame(scatter["scatter"])
    fig = px.scatter(df_scat, x="Frequency", y="Monetary", color=df_scat["Cluster"].astype(str))
    st.plotly_chart(fig)

st.subheader("Cluster Summary Table")
summary = get_cluster_summary()
if "clusters" in summary:
    df_sum = pd.DataFrame(summary["clusters"])
    st.dataframe(df_sum)
    st.download_button("📥 Download Clustered Data CSV", df_sum.to_csv(index=False), "cluster_summary.csv")

# Prediction
st.header("4. Predict Customer Segment")
with st.form("predict_form"):
    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (purchases)", min_value=1)
    monetary = st.number_input("Monetary ($)", min_value=0.0)
    clv = st.number_input("CLV ($)", min_value=0.0)
    submit = st.form_submit_button("Predict Segment")
    if submit:
        result = predict_cluster({"recency": recency, "frequency": frequency, "monetary": monetary, "clv": clv})
        if "predicted_cluster" in result:
            st.success(f"Predicted Cluster: {result['predicted_cluster']}")
