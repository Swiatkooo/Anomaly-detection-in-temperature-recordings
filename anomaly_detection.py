import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# === 1. Load data ===
df = pd.read_csv("dailymintemperatures.csv", on_bad_lines="skip")
df = df.rename(columns={df.columns[1]: "Temp"})
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
df = df.dropna(subset=["Temp"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Split: 1981–1987 training, 1988–1990 test
train_end = pd.Timestamp("1987-12-31")
train_df = df[df["Date"] <= train_end].copy()
test_df  = df[df["Date"] >  train_end].copy()

# Normalize based on training
tmin, tmax = train_df["Temp"].min(), train_df["Temp"].max()
train_df["TempN"] = (train_df["Temp"] - tmin) / (tmax - tmin)
test_df["TempN"]  = (test_df["Temp"]  - tmin) / (tmax - tmin)

# === 2. Prepare 7-day windows ===
WINDOW = 7
def make_windows(series, win=7):
    arr = series.values.astype(np.float32)
    X, Y = [], []
    for i in range(len(arr) - win + 1):
        seq = arr[i:i+win]
        X.append(seq)
        Y.append(seq)
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

trainX, trainY = make_windows(train_df["TempN"], WINDOW)
testX, testY   = make_windows(test_df["TempN"], WINDOW)

train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=32, shuffle=True)

# === 3. Define model ===
class AE(nn.Module):
    def __init__(self, win=7, latent=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(win, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, win), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(win=WINDOW, latent=5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

MODEL_FP32 = "anomaly_model_fp32.pth"
MODEL_INT8 = "anomaly_model_int8.pth"

# === 4. Load or train ===
if os.path.exists(MODEL_FP32) and os.path.exists(MODEL_INT8):
    print("Loaded saved FP32 and INT8 models.")
else:
    print("No saved models – starting training.")
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}")

    # save FP32 (state_dict)
    torch.save({
        "model_state": model.state_dict(),
        "tmin": float(tmin),
        "tmax": float(tmax)
    }, MODEL_FP32)
    print(f"FP32 model saved to {MODEL_FP32}")

    # quantize INT8 and save whole object
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save({
        "model": quantized_model,
        "tmin": float(tmin),
        "tmax": float(tmax)
    }, MODEL_INT8)
    print(f"INT8 model saved to {MODEL_INT8}")

# === 5. Testing on 1988–1990 data ===
def evaluate(model, data, thr_q=0.95):
    model.eval()
    with torch.no_grad():
        recon = model(data)
        errors = torch.mean((recon - data)**2, dim=1)
    thr = float(torch.quantile(errors, q=thr_q))
    flags = errors > thr
    return errors, thr, flags

# FP32
checkpoint_fp32 = torch.load(MODEL_FP32, weights_only=False)
model_fp32 = AE(win=WINDOW, latent=5)
model_fp32.load_state_dict(checkpoint_fp32["model_state"])
errors_fp32, thr_fp32, flags_fp32 = evaluate(model_fp32, testX)
print(f"[FP32] Detected {flags_fp32.sum().item()} anomalies out of {len(flags_fp32)} test windows.")

# INT8
checkpoint_int8 = torch.load(MODEL_INT8, weights_only=False)
model_int8 = checkpoint_int8["model"]   # whole object
errors_int8, thr_int8, flags_int8 = evaluate(model_int8, testX)
print(f"[INT8] Detected {flags_int8.sum().item()} anomalies out of {len(flags_int8)} test windows.")

# === 6. FP32 plots ===
plt.figure(figsize=(12,6))
plt.plot(errors_fp32.numpy(), label="Reconstruction error (MSE)", color="blue")
plt.axhline(y=thr_fp32, color="red", linestyle="--", label="Threshold")
plt.scatter(
    [i for i, f in enumerate(flags_fp32) if f],
    [errors_fp32[i].item() for i, f in enumerate(flags_fp32) if f],
    color="yellow", label="Anomalies"
)
plt.title("Anomaly detection (FP32 model)")
plt.xlabel("Window index (7-day sequences)")
plt.ylabel("Reconstruction error")
plt.legend()
plt.show()

# === 7. INT8 plots ===
plt.figure(figsize=(12,6))
plt.plot(errors_int8.numpy(), label="Reconstruction error (MSE)", color="blue")
plt.axhline(y=thr_int8, color="red", linestyle="--", label="Threshold")
plt.scatter(
    [i for i, f in enumerate(flags_int8) if f],
    [errors_int8[i].item() for i, f in enumerate(flags_int8) if f],
    color="yellow", label="Anomalies"
)
plt.title("Anomaly detection (INT8 model)")
plt.xlabel("Window index (7-day sequences)")
plt.ylabel("Reconstruction error")
plt.legend()
plt.show()
