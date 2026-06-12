#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:21:41 2025

@author: eta

Dimension reduction of dataset using autoencoder

# save 1/10 of each output in csv files
# use 1 gpu - a huge speed up
# optimize memory use
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------
# Config
# ------------------
INPUT_FILE = "/scratch/eta/ml_training_data_merged.csv"
CHUNK_COUNT = 10       # split into 10 chunks for output generation
BATCH_SIZE = 1024
EPOCHS = 20
BOTTLENECK_DIM = 50   # number of compressed dimensions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------
# Load column names
# ------------------
df_sample = pd.read_csv(INPUT_FILE, nrows=5)  # read small sample
chemberta_cols = [c for c in df_sample.columns if c.startswith("drug_")]
protbert_cols = [c for c in df_sample.columns if c.startswith("prot_")]

# ------------------
# Model
# ------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=50):   # model structure
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),   # introduce non-linearity
            nn.Linear(256, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),   # introduce non-linearity
            nn.Linear(256, input_dim)
        )

    def forward(self, x):   # model training
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ------------------
# Training function
# ------------------
def train_autoencoder_csv(input_file, cols, input_dim):
    # Read all relevant columns in chunks for training
    # make dataloader object
    df = pd.read_csv(input_file, usecols=cols)
    X = torch.tensor(df.values, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # specify model, optimizer, loss function, scaler
    model = Autoencoder(input_dim, BOTTLENECK_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # train autoencoder for 20 epochs, forward/backward pass, update parameters, iterating until the loss decreases
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                recon, _ = model(batch)
                loss = loss_fn(recon, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(loader.dataset):.6f}")
    return model

# ------------------
# Train both models separately and save models
# ------------------
chem_model = train_autoencoder_csv(INPUT_FILE, chemberta_cols, 768)
prot_model = train_autoencoder_csv(INPUT_FILE, protbert_cols, 1024)

# ------------------
# Feature generation in chunks (1/10 of whole data - optimize memory usage and running time)
# ------------------
def generate_features_in_chunks(input_file, cols, model, prefix, chunk_count):
    total_rows = sum(1 for _ in open(input_file)) - 1  # minus header
    chunk_size = total_rows // chunk_count
    results = []

    for chunk_idx, chunk_df in enumerate(pd.read_csv(input_file, usecols=cols, chunksize=chunk_size)):
        X_chunk = torch.tensor(chunk_df.values, dtype=torch.float32).to(DEVICE)
        reduced_list = []
        
        # get z, which is the compressed data from the model, and append it to list
        loader = DataLoader(TensorDataset(X_chunk), batch_size=BATCH_SIZE)
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                _, z = model(batch)
                reduced_list.append(z.cpu().numpy())
        
        # save the results
        reduced_chunk = np.vstack(reduced_list)
        results.append(pd.DataFrame(reduced_chunk, columns=[f"{prefix}_{i}" for i in range(BOTTLENECK_DIM)]))
        print(f"Processed chunk {chunk_idx+1}/{chunk_count}")
    return pd.concat(results, ignore_index=True)

# Generate reduced features
chemberta_reduced = generate_features_in_chunks(INPUT_FILE, chemberta_cols, chem_model, "chemberta_ae", CHUNK_COUNT)
protbert_reduced  = generate_features_in_chunks(INPUT_FILE, protbert_cols, prot_model, "protbert_ae", CHUNK_COUNT)

# ------------------
# Final merge and save
# ------------------
targets = pd.read_csv(INPUT_FILE, usecols=['ic50_nM'])
df_final = pd.concat([chemberta_reduced, protbert_reduced, targets], axis=1)
df_final.to_csv('/scratch/eta/ml_training_data_ae_reduced.csv', index=False)
print("Saved ml_training_data_ae_reduced.csv")

# the loss has decreased significantly
'''
Epoch 1/20, Loss=0.123739
Epoch 2/20, Loss=0.092392
Epoch 3/20, Loss=0.088990
Epoch 4/20, Loss=0.086636
Epoch 5/20, Loss=0.084665
Epoch 6/20, Loss=0.083131
Epoch 7/20, Loss=0.081971
Epoch 8/20, Loss=0.080984
Epoch 9/20, Loss=0.080172
Epoch 10/20, Loss=0.079478
Epoch 11/20, Loss=0.078933
Epoch 12/20, Loss=0.078488
Epoch 13/20, Loss=0.078096
Epoch 14/20, Loss=0.077764
Epoch 15/20, Loss=0.077476
Epoch 16/20, Loss=0.077206
Epoch 17/20, Loss=0.076963
Epoch 18/20, Loss=0.076742
Epoch 19/20, Loss=0.076531
Epoch 20/20, Loss=0.076339
Epoch 1/20, Loss=0.000845
Epoch 2/20, Loss=0.000181
Epoch 3/20, Loss=0.000134
Epoch 4/20, Loss=0.000108
Epoch 5/20, Loss=0.000091
Epoch 6/20, Loss=0.000078
Epoch 7/20, Loss=0.000069
Epoch 8/20, Loss=0.000063
Epoch 9/20, Loss=0.000059
Epoch 10/20, Loss=0.000056
Epoch 11/20, Loss=0.000053
Epoch 12/20, Loss=0.000051
Epoch 13/20, Loss=0.000049
Epoch 14/20, Loss=0.000048
Epoch 15/20, Loss=0.000047
Epoch 16/20, Loss=0.000046
Epoch 17/20, Loss=0.000045
Epoch 18/20, Loss=0.000044
Epoch 19/20, Loss=0.000043
Epoch 20/20, Loss=0.000042
'''