import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

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

# Print dataset info
print(f"Dataset Shape: {df.shape}")
print(df.head())

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict anomalies
df["Anomaly_Score"] = iso_forest.decision_function(X)
df["Fraud_Prediction"] = iso_forest.predict(X)

# Convert -1 to 1 (fraud), 1 to 0 (normal)
df["Fraud_Prediction"] = df["Fraud_Prediction"].apply(lambda x: 1 if x == -1 else 0)

# Save results locally
df.to_csv("processed_fraud_detection.csv", index=False)
print("Results saved locally.")

# Measure Processing Time vs. Dataset Size
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
plt.savefig("processing_time_plot.png")  # Save plot
print("Plot saved.")

# 🔽 Upload to Google Drive 🔽
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Authenticate
drive = GoogleDrive(gauth)

folder_id = "1Wbto677ngmBFo9fqsAhCUQbaw0Ydj1AF"  # Replace with your Google Drive Folder ID

def upload_to_drive(file_name):
    file_drive = drive.CreateFile({"title": file_name, "parents": [{"id": folder_id}]})
    file_drive.SetContentFile(file_name)
    file_drive.Upload()
    print(f"{file_name} uploaded to Google Drive.")

upload_to_drive("processed_fraud_detection.csv")
upload_to_drive("processing_time_plot.png")
