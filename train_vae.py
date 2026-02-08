# train_vae.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

# ==========================
# 1. CONFIG
# ==========================

LATENT_DIM = 16
BATCH_SIZE = 128
EPOCHS = 15
BETA = 1.0
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# 2. DATASET (MNIST)
# Normal = digit 0
# Anomaly = other digits
# ==========================

transform = transforms.Compose([
    transforms.ToTensor()
])

mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

X = []
y = []

for img, label in mnist:
    X.append(img.view(-1).numpy())
    y.append(0 if label == 0 else 1)  # 0 = normal, 1 = anomaly

X = np.array(X)
y = np.array(y)

# Train only on normal samples
X_train = X[y == 0]
X_test, y_test = X, y

train_loader = torch.utils.data.DataLoader(
    torch.tensor(X_train, dtype=torch.float32),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================
# 3. VAE MODEL
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
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ==========================
# 4. LOSS FUNCTION
# ==========================

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl

# ==========================
# 5. TRAINING
# ==========================

model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ==========================
# 6. ANOMALY DETECTION
# ==========================

model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    recon, mu, logvar = model(X_test_tensor)
    errors = torch.mean((recon - X_test_tensor) ** 2, dim=1)
    errors = errors.cpu().numpy()

# Threshold = 95th percentile of normal training errors
train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    recon_train, _, _ = model(train_tensor)
    train_errors = torch.mean((recon_train - train_tensor) ** 2, dim=1)
    threshold = np.percentile(train_errors.cpu().numpy(), 95)

print("Anomaly Threshold:", threshold)

y_pred = (errors > threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
