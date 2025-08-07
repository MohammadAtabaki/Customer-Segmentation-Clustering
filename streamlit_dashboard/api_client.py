import requests

BASE_URL = "http://localhost:8000"  # Adjust if API is remote

def upload_file(file):
    files = {"file": (file.name, file, "multipart/form-data")} if file else {}
    return requests.post(f"{BASE_URL}/upload", files=files).json()

def train_model(method="kmeans"):
    return requests.post(f"{BASE_URL}/train", params={"method": method}).json()


def predict_cluster(data):
    return requests.post(f"{BASE_URL}/predict", json=data).json()

def get_histograms():
    return requests.get(f"{BASE_URL}/visualization/rfm-histograms").json()

def get_cluster_summary():
    return requests.get(f"{BASE_URL}/visualization/cluster-summary").json()

def get_silhouette_curve():
    return requests.get(f"{BASE_URL}/visualization/silhouette-curve").json()

def get_scatter_plot():
    return requests.get(f"{BASE_URL}/visualization/segmentation-scatter").json()

def get_csv_download_url():
    return f"{BASE_URL}/visualization/download/csv"

def get_model_download_url():
    return f"{BASE_URL}/visualization/download/model"