import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Google Drive Authentication
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

# Print results
print(df[["Class", "Fraud_Prediction"]].head(20))

# Save processed results
df.to_csv("processed_fraud_detection.csv", index=False)
print("Results saved locally.")

# Benchmark processing time
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
plt.savefig("processing_time_plot.png")  # Save the plot
plt.show()
print("Plot saved.")

# ---------- GOOGLE DRIVE UPLOAD ------------
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Authenticate user

drive = GoogleDrive(gauth)

# Folder ID (replace with your actual Google Drive folder ID)
FOLDER_ID = "1Wbto677ngmBFo9fqsAhCUQbaw0Ydj1AF"  # If empty, uploads to My Drive

# Upload file
file_metadata = {"title": "processing_time_plot.png"}
if FOLDER_ID:
    file_metadata["parents"] = [{"id": FOLDER_ID}]

file = drive.CreateFile(file_metadata)
file.SetContentFile("processing_time_plot.png")
file.Upload()

print(f"File uploaded successfully! View at: https://drive.google.com/file/d/{file['id']}/view")

