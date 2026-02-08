import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ===============================
# CONFIG
# ===============================

LATENT_OPTIONS = [8, 16, 32]
BETA_OPTIONS = [0.5, 1.0, 2.0]
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ===============================
# HIGH-DIMENSION DATASET (120 features)
# ===============================

X, y = make_classification(
    n_samples=5000,
    n_features=120,
    n_informative=40,
    n_redundant=20,
    n_classes=2,
    random_state=42
)

# Treat class 0 as normal, class 1 as anomaly
y_binary = y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_normal = X_scaled[y_binary == 0]
X_anomaly = X_scaled[y_binary == 1]

X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

X_test = X_scaled
y_test = y_binary

input_dim = X_scaled.shape[1]
print("Feature Dimension:", input_dim)

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# ===============================
# MODEL
# ===============================

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


# ===============================
# GRID SEARCH
# ===============================

best_f1 = 0
best_latent = None
best_beta = None

for latent_dim in LATENT_OPTIONS:
    for beta in BETA_OPTIONS:

        print(f"\nTesting latent={latent_dim}, beta={beta}")

        model = VAE(input_dim, latent_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        def loss_fn(recon_x, x, mu, logvar):
            recon_loss = nn.functional.mse_loss(recon_x, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + beta * kl

        # Train
        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                loss = loss_fn(recon, batch, mu, logvar)
                loss.backward()
                optimizer.step()

        # Validation threshold
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

        print("F1:", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_latent = latent_dim
            best_beta = beta

print("\nBest Latent:", best_latent)
print("Best Beta:", best_beta)
print("Best VAE F1:", best_f1)

# ===============================
# Train Baseline AE with same threshold logic
# ===============================

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
    recon_val = ae(val_tensor)
    val_errors = torch.mean((recon_val - val_tensor)**2, dim=1).cpu().numpy()

threshold_ae = np.percentile(val_errors, 95)

with torch.no_grad():
    recon_test = ae(test_tensor)
    test_errors = torch.mean((recon_test - test_tensor)**2, dim=1).cpu().numpy()

ae_preds = (test_errors > threshold_ae).astype(int)

ae_f1 = f1_score(y_test, ae_preds)

print("\nBaseline AE F1:", ae_f1)
print("VAE Threshold:", threshold)
print("AE Threshold:", threshold_ae)
