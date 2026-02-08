import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import datasets, transforms

# ==========================================================
# CONFIGURATION
# ==========================================================

LATENT_DIM = 16
BETA = 1.0
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ==========================================================
# DATASET PREPARATION (MNIST)
# Normal = digit 0
# Anomaly = digits 1-9
# ==========================================================

transform = transforms.ToTensor()

mnist = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

X = []
y = []

for img, label in mnist:
    X.append(img.view(-1).numpy())
    y.append(0 if label == 0 else 1)  # 0 normal, 1 anomaly

X = np.array(X)
y = np.array(y)

# Train only on normal samples
X_train = X[y == 0]
X_test = X
y_test = y

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

# ==========================================================
# VAE MODEL
# ==========================================================

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )

        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
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
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl


# ==========================================================
# BASELINE AUTOENCODER
# ==========================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=LATENT_DIM):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ==========================================================
# TRAIN VAE
# ==========================================================

vae_model = VAE().to(DEVICE)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=LR)

print("\nTraining VAE...")

for epoch in range(EPOCHS):
    vae_model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        vae_optimizer.zero_grad()
        recon, mu, logvar = vae_model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        vae_optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")


# ==========================================================
# TRAIN BASELINE AUTOENCODER
# ==========================================================

ae_model = Autoencoder().to(DEVICE)
ae_optimizer = optim.Adam(ae_model.parameters(), lr=LR)

print("\nTraining Autoencoder...")

for epoch in range(EPOCHS):
    ae_model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        ae_optimizer.zero_grad()
        recon = ae_model(batch)
        loss = nn.functional.mse_loss(recon, batch)
        loss.backward()
        ae_optimizer.step()

        total_loss += loss.item()

    print(f"[AE] Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")


# ==========================================================
# EVALUATE VAE
# ==========================================================

vae_model.eval()

with torch.no_grad():
    recon_test, _, _ = vae_model(X_test_tensor)
    errors = torch.mean((recon_test - X_test_tensor) ** 2, dim=1)
    errors = errors.cpu().numpy()

with torch.no_grad():
    recon_train, _, _ = vae_model(train_tensor)
    train_errors = torch.mean((recon_train - train_tensor) ** 2, dim=1)
    threshold = np.percentile(train_errors.cpu().numpy(), 95)

vae_pred = (errors > threshold).astype(int)

vae_precision = precision_score(y_test, vae_pred)
vae_recall = recall_score(y_test, vae_pred)
vae_f1 = f1_score(y_test, vae_pred)

# ==========================================================
# EVALUATE AUTOENCODER
# ==========================================================

ae_model.eval()

with torch.no_grad():
    recon_test_ae = ae_model(X_test_tensor)
    ae_errors = torch.mean((recon_test_ae - X_test_tensor) ** 2, dim=1)
    ae_errors = ae_errors.cpu().numpy()

with torch.no_grad():
    recon_train_ae = ae_model(train_tensor)
    ae_train_errors = torch.mean((recon_train_ae - train_tensor) ** 2, dim=1)
    ae_threshold = np.percentile(ae_train_errors.cpu().numpy(), 95)

ae_pred = (ae_errors > ae_threshold).astype(int)

ae_precision = precision_score(y_test, ae_pred)
ae_recall = recall_score(y_test, ae_pred)
ae_f1 = f1_score(y_test, ae_pred)

# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n================ FINAL RESULTS ================")

print("\nVAE Results:")
print("Threshold:", threshold)
print("Precision:", vae_precision)
print("Recall:", vae_recall)
print("F1 Score:", vae_f1)

print("\nBaseline Autoencoder Results:")
print("Threshold:", ae_threshold)
print("Precision:", ae_precision)
print("Recall:", ae_recall)
print("F1 Score:", ae_f1)
