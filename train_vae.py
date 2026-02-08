import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ================================
# CONFIG
# ================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
PERCENTILE = 95

LATENT_DIMS = [8, 16, 32]
BETAS = [0.5, 1.0, 2.0]

print("Device:", DEVICE)

# ================================
# DATASET (MNIST)
# ================================

transform = transforms.ToTensor()
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)

X = []
y = []

for img, label in mnist:
    X.append(img.view(-1).numpy())
    y.append(0 if label == 0 else 1)  # 0 normal, 1 anomaly

X = np.array(X)
y = np.array(y)

# Train / Val / Test
X_normal = X[y == 0]

X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
X_test, y_test = X, y

# ================================
# MODELS
# ================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ================================
# GRID SEARCH (VALIDATION)
# ================================

best_score = 0
best_config = None

for latent in LATENT_DIMS:
    for beta in BETAS:

        model = VAE(784, latent).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR)

        train_loader = torch.utils.data.DataLoader(
            torch.tensor(X_train, dtype=torch.float32),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Train
        for _ in range(EPOCHS):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                opt.zero_grad()
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar, beta)
                loss.backward()
                opt.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            recon, _, _ = model(val_tensor)
            errors = torch.mean((recon - val_tensor) ** 2, dim=1).cpu().numpy()

        threshold = np.percentile(errors, PERCENTILE)
        preds = (errors > threshold).astype(int)

        score = -np.mean(errors)  # proxy score
        print(f"Latent={latent}, Beta={beta}, Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_config = (latent, beta)

print("\nBEST CONFIG:", best_config)

# ================================
# FINAL TRAINING
# ================================

LATENT_DIM, BETA = best_config

vae = VAE(784, LATENT_DIM).to(DEVICE)
ae = Autoencoder(784, LATENT_DIM).to(DEVICE)

opt_vae = optim.Adam(vae.parameters(), lr=LR)
opt_ae = optim.Adam(ae.parameters(), lr=LR)

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

for _ in range(EPOCHS):
    for batch in train_loader:
        batch = batch.to(DEVICE)

        # VAE
        opt_vae.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar, BETA)
        loss.backward()
        opt_vae.step()

        # AE
        opt_ae.zero_grad()
        recon_ae = ae(batch)
        loss_ae = nn.functional.mse_loss(recon_ae, batch)
        loss_ae.backward()
        opt_ae.step()

# ================================
# TEST EVALUATION
# ================================

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    recon, _, _ = vae(X_test_tensor)
    err_vae = torch.mean((recon - X_test_tensor) ** 2, dim=1).cpu().numpy()

    recon_ae = ae(X_test_tensor)
    err_ae = torch.mean((recon_ae - X_test_tensor) ** 2, dim=1).cpu().numpy()

thr_vae = np.percentile(err_vae[y_test == 0], PERCENTILE)
thr_ae = np.percentile(err_ae[y_test == 0], PERCENTILE)

pred_vae = (err_vae > thr_vae).astype(int)
pred_ae = (err_ae > thr_ae).astype(int)

print("\n==== FINAL METRICS ====")

print("\nVAE")
print("Precision:", precision_score(y_test, pred_vae))
print("Recall:", recall_score(y_test, pred_vae))
print("F1:", f1_score(y_test, pred_vae))

print("\nAutoencoder")
print("Precision:", precision_score(y_test, pred_ae))
print("Recall:", recall_score(y_test, pred_ae))
print("F1:", f1_score(y_test, pred_ae))
