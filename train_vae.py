import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================================
# CONFIG
# ==========================================================

LATENT_OPTIONS = [4, 8, 16]
BETA = 1.0
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ==========================================================
# LOAD DATASET
# ==========================================================

data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign (in sklearn)

# We treat BENIGN as normal (1), MALIGNANT as anomaly (0)
y_binary = np.where(y == 1, 0, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_normal = X_scaled[y_binary == 0]
X_anomaly = X_scaled[y_binary == 1]

X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

X_test = X_scaled
y_test = y_binary

input_dim = X_scaled.shape[1]
print("Input dimension:", input_dim)

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
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
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
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl


# ==========================================================
# HYPERPARAMETER SEARCH
# ==========================================================

best_latent = None
best_f1 = 0
search_results = {}

for latent_dim in LATENT_OPTIONS:
    print("\nTesting Latent Dimension:", latent_dim)

    model = VAE(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        recon_val, _, _ = model(val_tensor)
        val_errors = torch.mean((recon_val - val_tensor)**2, dim=1).cpu().numpy()

    threshold = np.percentile(val_errors, 95)

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

    search_results[latent_dim] = f1

    if f1 > best_f1:
        best_f1 = f1
        best_latent = latent_dim

print("\nBest Latent Dimension:", best_latent)

# ==========================================================
# TRAIN BASELINE AUTOENCODER
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
    recon = ae(test_tensor)
    errors = torch.mean((recon - test_tensor)**2, dim=1).cpu().numpy()

threshold_ae = np.percentile(errors[y_test == 0], 95)
ae_preds = (errors > threshold_ae).astype(int)

ae_precision = precision_score(y_test, ae_preds)
ae_recall = recall_score(y_test, ae_preds)
ae_f1 = f1_score(y_test, ae_preds)

print("\n================ FINAL SUMMARY ================")
print("Best Latent Dimension:", best_latent)
print("Best VAE F1:", best_f1)
print("Baseline AE F1:", ae_f1)
print("VAE Threshold (95th percentile):", threshold)
print("AE Threshold (95th percentile):", threshold_ae)
