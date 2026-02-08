import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ==========================================================
# CONFIG
# ==========================================================

BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
BETA = 1.0
LATENT_OPTIONS = [8, 16, 32]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ==========================================================
# LOAD DATA
# ==========================================================

data = fetch_kddcup99(subset="SA", percent10=True)

X_raw = pd.DataFrame(data.data)
y_raw = pd.Series(data.target)

# Decode labels
y_raw = y_raw.apply(lambda x: x.decode())
y_binary = (y_raw != "normal.").astype(int)

# Identify categorical columns (object type)
categorical_cols = X_raw.select_dtypes(include=["object"]).columns
numeric_cols = X_raw.select_dtypes(exclude=["object"]).columns

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[categorical_cols])

# Scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(X_raw[numeric_cols])

# Combine
X_processed = np.hstack([X_num, X_cat])

# Split normal vs anomaly
X_normal = X_processed[y_binary == 0]
X_anomaly = X_processed[y_binary == 1]

# Train/Validation split (only normal data for training)
X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

# Full test set (normal + anomaly)
X_test = X_processed
y_test = y_binary.values

input_dim = X_processed.shape[1]

print("Final feature dimension:", input_dim)

# ==========================================================
# DATA LOADERS
# ==========================================================

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# ==========================================================
# MODEL DEFINITIONS
# ==========================================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl


# ==========================================================
# HYPERPARAMETER SEARCH
# ==========================================================

best_latent = None
best_f1 = 0
results = {}

for latent_dim in LATENT_OPTIONS:
    print("\nTesting Latent Dimension:", latent_dim)

    model = VAE(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()

    # Validation threshold
    model.eval()
    with torch.no_grad():
        recon_val, _, _ = model(val_tensor)
        val_errors = torch.mean((recon_val - val_tensor)**2, dim=1).cpu().numpy()

    threshold = np.percentile(val_errors, 95)

    # Test evaluation
    with torch.no_grad():
        recon_test, _, _ = model(test_tensor)
        test_errors = torch.mean((recon_test - test_tensor)**2, dim=1).cpu().numpy()

    preds = (test_errors > threshold).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    results[latent_dim] = f1

    if f1 > best_f1:
        best_f1 = f1
        best_latent = latent_dim

print("\nBest Latent Dimension:", best_latent)

# ==========================================================
# TRAIN BASELINE AE WITH BEST LATENT
# ==========================================================

ae = Autoencoder(input_dim, best_latent).to(DEVICE)
optimizer = optim.Adam(ae.parameters(), lr=LR)

for epoch in range(EPOCHS):
    ae.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon = ae(batch)
        loss = nn.functional.mse_loss(recon, batch)
        loss.backward()
        optimizer.step()

ae.eval()
with torch.no_grad():
    recon_test = ae(test_tensor)
    errors = torch.mean((recon_test - test_tensor)**2, dim=1).cpu().numpy()

threshold = np.percentile(errors[y_test == 0], 95)
ae_preds = (errors > threshold).astype(int)

ae_precision = precision_score(y_test, ae_preds)
ae_recall = recall_score(y_test, ae_preds)
ae_f1 = f1_score(y_test, ae_preds)

print("\n================ FINAL RESULTS ================")
print("Best VAE Latent:", best_latent)
print("Best VAE F1:", best_f1)
print("\nAutoencoder Results:")
print("Precision:", ae_precision)
print("Recall:", ae_recall)
print("F1:", ae_f1)
