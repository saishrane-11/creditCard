import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from flask import Flask

# Initialize Flask app (for Render deployment)
app = Flask(__name__)

# Load dataset
df = pd.read_csv("creditcard.csv")

# Normalize 'Amount' feature
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

# Drop unnecessary columns
df = df.drop(["Time"], axis=1)

# Separate features and labels
X = df.drop("Class", axis=1)  # Features
y = df["Class"]  # Labels (0 = normal, 1 = fraud)

# Train Isolation Forest model
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict anomalies
df["Anomaly_Score"] = iso_forest.decision_function(X)
df["Fraud_Prediction"] = iso_forest.predict(X)

# Convert -1 to 1 (fraud), 1 to 0 (normal)
df["Fraud_Prediction"] = df["Fraud_Prediction"].apply(lambda x: 1 if x == -1 else 0)

# Save results
df.to_csv("processed_fraud_detection.csv", index=False)

# Measure processing time for different dataset sizes
sizes = [1000, 5000, 10000, 20000, 50000]
times = []

for size in sizes:
    subset = X[:size]

    start_time = time.time()
    iso_forest.fit(subset)  # Train model
    end_time = time.time()

    times.append(end_time - start_time)

# Plot results
plt.plot(sizes, times, marker="o")
plt.xlabel("Dataset Size")
plt.ylabel("Processing Time (seconds)")
plt.title("Processing Time vs. Dataset Size")
plt.savefig("fraud_detection_plot.png")  # Save as PNG

print("Plot saved successfully!")

# =======================
# Upload to Google Drive
# =======================
def upload_to_drive(filename):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Authenticate
    drive = GoogleDrive(gauth)

    file = drive.CreateFile({'title': filename})  # Upload file
    file.SetContentFile(filename)
    file.Upload()

    print(f"Uploaded {filename} to Google Drive!")

upload_to_drive("fraud_detection_plot.png")

# =======================
# Flask Route (Render)
# =======================
@app.route("/")
def home():
    return "Fraud detection model is running and files are uploaded to Google Drive!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render requires PORT variable
    app.run(host="0.0.0.0", port=port)

