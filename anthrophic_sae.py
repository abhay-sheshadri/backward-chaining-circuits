import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader


class AnthrophicSparseAutoEncoder(nn.Module):
    """
        This class implements a sparse autoencoder as described by Anthropic (https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder).
    """

    def __init__(self, n, m, l1_coefficient=0.0001):
        super(AnthrophicSparseAutoEncoder, self).__init__()
        self.n = n  # input and output dimension
        self.m = m  # autoencoder hidden layer dimension (usually m >= n)

        # initialize encoder and decoder weights with Kaiming Uniform initialization
        self.W_e = nn.Parameter(torch.empty(self.m, self.n))
        self.W_d = nn.Parameter(torch.empty(self.n, self.m))
        nn.init.kaiming_uniform_(self.W_e, a=math.sqrt(5))  # we use a=math.sqrt(5) as this is the standard init for PyTorch's Linear 
        nn.init.kaiming_uniform_(self.W_d, a=math.sqrt(5))

        # initialize the biases
        self.b_e = nn.Parameter(torch.randn(m))
        self.b_d = nn.Parameter(torch.randn(n))

        self.l1_coefficient = l1_coefficient


    def forward(self, x):
        # normalize columns to have unit norm
        self.W_e = F.normalize(self.W_e, dim=1)  # not sure whether this is necessary
        self.W_d = F.normalize(self.W_d, dim=1)

        # subtract decoder bias ("pre-encoder bias")
        x_bar = x - self.b_d

        # encode
        f = F.relu(torch.matmul(self.W_e, x_bar) + self.b_e)

        # decode
        x_hat = torch.matmul(self.W_d, f) + self.b_d

        # compute loss
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        sparsity_loss = torch.norm(f, p=1)
        loss = reconstruction_loss + self.l1_coefficient * sparsity_loss
        return x_hat, f, loss


class AnthropicSparseAutoEncoderTrainer:
    """
        This class trains the AnthropicSpareAutoEncoder given a dataset of activations.
    """

    def __init__(self, n, m, l1_coefficient, lr):
        self.sae = AnthrophicSparseAutoEncoder(n, m, l1_coefficient)
        self.lr = lr

        
    def fit(self, X, n_steps=1000, train_size=0.8):

        assert isinstance(X, torch.Tensor)

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.sae.parameters,
            lr = self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=n_steps,
            eta_min=2e-6
        )

        # dataset
        train_dataset, validation_dataset = random_split(X, [int(len(X) * train_size), int(len(X) * (1-train_size))])
        train_loader, validation_loader = DataLoader(train_dataset), DataLoader(validation_dataset)
        for step in range(n_steps):

            # training
            training_loss = 0
            for batch in train_loader:
                batch = batch[0].to(self.device)
                self.optimizer.zero_grad()
                _, _, loss = self.model(batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()

            # evaluation
            self.model.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch in validation_loader:
                    batch = batch[0].to(self.device)
                    _, _, loss = self.model(batch)
                    val_loss += loss

            if self.verbose:
                print(f"Epoch {step} - Training Loss: {training_loss:.4f} - Validation Loss: {validation_loss:.4f}")
            

            


if __name__ == "__main__":
    
