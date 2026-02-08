import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ================================
# CONFIG
# ================================

LATENT_DIM = 16
BETA = 1.0
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ================================
# LOAD KDD DATASET
# ================================

data = fetch_kddcup99(subset='SA', percent10=True)

X = data.data
y = data.target

# Convert bytes to string labels
y = np.array([label.decode() for label in y])

# Normal = "normal"
y_binary = np.array([0 if label == "normal." else 1 for label in y])

# Convert categorical features to numeric
X = np.array(X)

# Keep only numeric columns
numeric_mask = []
for i in range(X.shape[1]):
    try:
        X[:, i].astype(float)
        numeric_mask.append(i)
    except:
        continue

X = X[:, numeric_mask].astype(float)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train only on normal data
X_train = X[y_binary == 0]
X_test = X
y_test = y_binary

input_dim = X.shape[1]

# ================================
# DATALOADER
# ================================

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

# ================================
# VAE MODEL
# ================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )

        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
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


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl


# ================================
# BASELINE AUTOENCODER
# ================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ================================
# TRAIN VAE
# ================================

vae = VAE(input_dim, LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=LR)

print("\nTraining VAE...")
for epoch in range(EPOCHS):
    vae.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


# ================================
# TRAIN AUTOENCODER
# ================================

ae = Autoencoder(input_dim, LATENT_DIM).to(DEVICE)
optimizer_ae = optim.Adam(ae.parameters(), lr=LR)

print("\nTraining Autoencoder...")
for epoch in range(EPOCHS):
    ae.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer_ae.zero_grad()
        recon = ae(batch)
        loss = nn.functional.mse_loss(recon, batch)
        loss.backward()
        optimizer_ae.step()
        total_loss += loss.item()

    print(f"[AE] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


# ================================
# EVALUATION
# ================================

vae.eval()
with torch.no_grad():
    recon, _, _ = vae(X_test_tensor)
    errors = torch.mean((recon - X_test_tensor)**2, dim=1).cpu().numpy()

threshold = np.percentile(errors[y_test == 0], 95)
vae_pred = (errors > threshold).astype(int)

vae_precision = precision_score(y_test, vae_pred)
vae_recall = recall_score(y_test, vae_pred)
vae_f1 = f1_score(y_test, vae_pred)


ae.eval()
with torch.no_grad():
    recon_ae = ae(X_test_tensor)
    ae_errors = torch.mean((recon_ae - X_test_tensor)**2, dim=1).cpu().numpy()

ae_threshold = np.percentile(ae_errors[y_test == 0], 95)
ae_pred = (ae_errors > ae_threshold).astype(int)

ae_precision = precision_score(y_test, ae_pred)
ae_recall = recall_score(y_test, ae_pred)
ae_f1 = f1_score(y_test, ae_pred)

print("\n=========== FINAL RESULTS ===========")
print("\nVAE")
print("Precision:", vae_precision)
print("Recall:", vae_recall)
print("F1:", vae_f1)

print("\nAutoencoder")
print("Precision:", ae_precision)
print("Recall:", ae_recall)
print("F1:", ae_f1)
