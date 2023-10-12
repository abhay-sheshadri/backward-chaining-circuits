import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import torch.nn.functional as F
import math


class SparseAutoencoder(nn.Module):
    """
        This class implements a sparse autoencoder as described by Anthropic:
        (https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder)
    """

    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        print("this is the Anthropic SAE")
        self.input_size = input_size  # input and output dimension
        self.hidden_size = hidden_size  # autoencoder hidden layer dimension (usually m >= n)

        # initialize encoder and decoder weights with Kaiming Uniform initialization
        self.W_e = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_d = nn.Parameter(torch.randn(input_size, hidden_size))
        nn.init.kaiming_uniform_(self.W_e, a=math.sqrt(5))  # we use a=math.sqrt(5) as this is the standard init for PyTorch's Linear 
        nn.init.kaiming_uniform_(self.W_d, a=math.sqrt(5))

        # initialize the biases
        self.b_e = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        # normalize columns to have unit norm
        W_e = F.normalize(self.W_e, dim=1, p=2)
        W_d = F.normalize(self.W_d, dim=1, p=2)
        # subtract decoder bias ("pre-encoder bias")
        x_bar = x - self.b_d
        # Encode into features
        features = F.relu(torch.matmul(x_bar, W_e.T) + self.b_e)
        # Reconstruct with decoder
        reconstruction = torch.matmul(features, W_d.T) + self.b_d
        return features, reconstruction


class SparseCoder:

    def __init__(
        self,
        num_codes: int,
        l1_coef: float = .00086,
        learning_rate_init: float = 1e-4,
        batch_size: int = 8192,
        max_iter: int = 10_000,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_codes = num_codes
        self.l1_coef = l1_coef
        self.model = None
        
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device

    def construct_model(self, input_dim):
        self.model = SparseAutoencoder(input_dim, self.num_codes)

    def fix_inputs(self, *args):
        # All inputs to tensors
        tensors = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                tensors.append(torch.from_numpy(arg))
            elif isinstance(arg, torch.Tensor):
                tensors.append(arg)
            else:
                raise ValueError("All input variables should be either torch tensors or numpy arrays.")
        if len(tensors) == 1:
            return tensors[0]
        else:
            return tuple(tensors)

    def get_loss(self, data, features, reconstruction):
        sparsity_loss = self.l1_coef * torch.norm(features, 1, dim=-1).mean()
        true_sparsity_loss = torch.norm(features, 0, dim=-1).mean()
        reconstruction_loss = F.mse_loss(reconstruction, data)
        return reconstruction_loss, sparsity_loss
    
    def get_recons_loss(self, data, features, reconstruction):
        reconstruction_loss = F.mse_loss(reconstruction, data)
        return reconstruction_loss

    def fit(self, X):

        def avg(lst): 
            return sum(lst) / len(lst) 

        X = self.fix_inputs(X)
        assert len(X.shape) == 2, f"X should have 2 dimensions, but has {len(X.shape)}"
        # Create model
        self.construct_model(X.shape[1])
        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate_init
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_iter,
            eta_min=2e-6
        )
        # Partition into training and validation set
        dataset = TensorDataset(X)
        train_size = int(len(dataset) * 0.9) # Use 10% of training data to validate
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,)
        # Start training and logging
        for epoch in range(self.max_iter):
            # Train model on training set
            self.model.train()  # set the model to training mode
            total_loss = 0

            sparsities = []
            reconstruction_losses = []
            sparsity_losses = []
            for bX in train_loader:
                bX = bX[0].to(self.device)
                self.optimizer.zero_grad()
                features, reconstruction = self.model(bX)
                reconstruction_loss, sparsity_loss = self.get_loss(bX, features, reconstruction)
                loss = reconstruction_loss + sparsity_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                sparsities.append((features != 0).float().mean(dim=0).sum().cpu().item())
                reconstruction_losses.append(reconstruction_loss)
                sparsity_losses.append(sparsity_loss)
            # self.scheduler.step()
            # Evaluate model on validation set
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for bX in val_loader:
                    bX = bX[0].to(self.device)  
                    features, reconstruction = self.model(bX)
                    val_loss += self.get_recons_loss(bX, features, reconstruction)
            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch} | Sparsity: {avg(sparsities):.4f} | Training Loss: {total_loss:.4f} | Validation Loss: {val_loss:.4f} | Reconstruction Loss: {avg(reconstruction_losses):.4f} | Sparsity Loss: {avg(sparsity_losses):.4f}")

    def featurize(self, X):
        assert self.model is not None, "Model has not been trained"
        X = self.fix_inputs(X)
        assert len(X.shape) == 2, f"X should have 2 dimensions, but has {len(X.shape)}"
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        outs = []
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for bX in loader:
                bX = bX[0].to(self.device)
                features, reconstruction = self.model(bX)
                outs.append(features.detach().cpu())
        return torch.cat(outs, dim=0)

    def reconstruct(self, X):
        assert self.model is not None, "Model has not been trained"
        X = self.fix_inputs(X)
        assert len(X.shape) == 2, f"X should have 2 dimensions, but has {len(X.shape)}"
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        outs = []
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for bX in loader:
                bX = bX[0].to(self.device)
                features, reconstruction = self.model(bX)
                outs.append(reconstruction.detach().cpu())
        return torch.cat(outs, dim=0)
