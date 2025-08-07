import subprocess
import time
import os
import sys

# Paths
BACKEND_DIR = os.path.join("customer-segmention")
FRONTEND_DIR = os.path.join("streamlit_dashboard")

# Start FastAPI backend
print("🚀 Starting FastAPI server...")
uvicorn_cmd = [
    sys.executable, "-m", "uvicorn",
    "app.api:app", "--reload"
]
backend_process = subprocess.Popen(uvicorn_cmd, cwd=BACKEND_DIR)

# Give it time to start
time.sleep(3)

# Start Streamlit frontend
print("🚀 Starting Streamlit dashboard...")
streamlit_cmd = [
    "streamlit", "run", "app.py"
]
subprocess.run(streamlit_cmd, cwd=FRONTEND_DIR)

# When Streamlit closes, kill the backend process
backend_process.terminate()
print("🛑 Backend process terminated.")
