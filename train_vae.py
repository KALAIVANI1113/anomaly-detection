import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import datasets, transforms

# ==========================
# CONFIG
# ==========================

LATENT_DIM = 16
BATCH_SIZE = 128
EPOCHS = 15
BETA = 1.0
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# DATASET (MNIST)
# ==========================

transform = transforms.ToTensor()
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

X = []
y = []

for img, label in mnist:
    X.append(img.view(-1).numpy())
    y.append(0 if label == 0 else 1)

X = np.array(X)
y = np.array(y)

X_train = X[y == 0]
X_test = X:
y_test = y

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

# ==========================
# VAE MODEL
# ==========================

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
        recon, mu, logvar = None, None, None
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl


# ==========================
# BASELINE AUTOENCODER
# ==========================

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
        return self.decoder(self.encoder(x))


# ==========================
# TRAIN VAE
# ==========================

vae = VAE().to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=LR)

for epoch in range(EPOCHS):
    vae.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()

# ==========================
# TRAIN AUTOENCODER
# ==========================

ae = Autoencoder().to(DEVICE)
optimizer_ae = optim.Adam(ae.parameters(), lr=LR)

for epoch in range(EPOCHS):
    ae.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer_ae.zero_grad()
        recon = ae(batch)
        loss = nn.functional.mse_loss(recon, batch)
        loss.backward()
        optimizer_ae.step()

# ==========================
# EVALUATION
# ==========================

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
    recon = ae(X_test_tensor)
    ae_errors = torch.mean((recon - X_test_tensor)**2, dim=1).cpu().numpy()

ae_threshold = np.percentile(ae_errors[y_test == 0], 95)
ae_pred = (ae_errors > ae_threshold).astype(int)

ae_precision = precision_score(y_test, ae_pred)
ae_recall = recall_score(y_test, ae_pred)
ae_f1 = f1_score(y_test, ae_pred)

print("\n===== FINAL RESULTS =====")
print("\nVAE:")
print("Precision:", vae_precision)
print("Recall:", vae_recall)
print("F1:", vae_f1)

print("\nAutoencoder:")
print("Precision:", ae_precision)
print("Recall:", ae_recall)
print("F1:", ae_f1)
