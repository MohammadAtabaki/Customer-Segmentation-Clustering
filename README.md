📊 Customer Segmentation with FastAPI & Streamlit
=================================================

A complete end-to-end application for **Customer Segmentation using RFM + CLV analysis**, powered by:

-   ⚙️ FastAPI (Backend APIs)

-   🎨 Streamlit (Interactive dashboard)

-   🤖 Multiple clustering algorithms: `KMeans`, `GMM`, `Hierarchical`, `DBSCAN`

-   📈 Visualization, interactivity, and CSV/model downloads

-   ✅ Modular codebase + unit tests

* * * * *

🚀 Overview
-----------

This project helps businesses understand their customers by automatically grouping them into meaningful **segments** using:

-   **RFM Analysis** (Recency, Frequency, Monetary)

-   **CLV** (Customer Lifetime Value)

Users can:

-   Upload their own customer dataset

-   Train a segmentation model (choose from 4 algorithms)

-   Explore results visually

-   Download clustered data and trained models

* * * * *

🧠 Core Logic
-------------

### 1\. **Preprocessing**

-   Clean data: remove nulls, filter bad transactions

-   Create `TotalPrice` = `Quantity` × `UnitPrice`

-   Prepare for RFM & CLV calculations

### 2\. **Feature Engineering**

-   **RFM features**:

    -   `Recency`: days since last purchase

    -   `Frequency`: number of invoices

    -   `Monetary`: total money spent

-   **CLV features**:

    -   `AvgOrderValue`, `CLV` = AOV × Frequency × Margin × Time

### 3\. **Clustering Algorithms**

-   `KMeans` with auto-`k` (silhouette-based)

-   `GMM` with automatic component selection

-   `Agglomerative` (hierarchical) clustering

-   `DBSCAN` (density-based, detects noise)

### 4\. **Evaluation**

-   Silhouette Score

-   Davies-Bouldin Score

-   Number of clusters

-   Cluster visualizations

### 5\. **Deployment Modes**

-   ✅ **Hybrid fallback**:

    -   If no dataset uploaded → use pre-trained model (KMeans)

    -   If dataset uploaded → train new model of selected type

* * * * *

💻 How to Use
-------------

### 1\. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### 2\. 🧠 Pretrain a model (Optional)

```bash
python app/utils/pretrain_model.py
```

### 3\. ⚙️ Run the project

```bash
python run_app.py
```

* * * * *

🧪 Testing
----------

To run unit tests:

```bash
cd customer-segmentation
pytest tests/`
```


Tests cover:

-   RFM and CLV feature generation

-   KMeans clustering with auto-k

-   Preprocessing logic

* * * * *

🧰 Key Features
---------------

| Feature | Description |
| --- | --- |
| 📥 Dataset upload | Upload `.csv` or `.xlsx` files with customer transactions |
| ⚙️ Train model | Choose method: `kmeans`, `gmm`, `hierarchical`, `dbscan` |
| 📊 Visualizations | Interactive scatter plots and silhouette curves |
| 📁 Download outputs | Export clustered dataset (`.csv`) and trained model (`.pkl`) |
| 🤖 Pretrained fallback | Uses pre-trained KMeans model if no dataset is uploaded |
| ✅ Unit tested | With `pytest`, test coverage for core logic |

* * * * *

📦 File Downloads
-----------------

After training:

-   ✅ **Download CSV** with cluster labels

-   ✅ **Download model** (`.pkl`) --- for `kmeans` and `gmm`

* * * * *

⚠️ Requirements for Custom Dataset
----------------------------------

Your dataset must include:

-   `InvoiceDate`

-   `InvoiceNo`

-   `CustomerID`

-   `Quantity`

-   `UnitPrice`

* * * * *

