import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class Probe:
    
    def __init__(
        self,
        learning_rate_init: float = 1e-3,
        batch_size: int = 2048,
        max_iter: int = 200,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.model = None
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device

    def construct_model(self, input_dim, output_dim):
        assert NotImplementedError("Model not defined")
        
    def save(self, filename):
        assert self.model is not None, "Model has not been trained"
        torch.save(self.model, filename)
        
    def load(self, filename):
        self.model = torch.load(filename)

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

    def get_loss(self, y, pred):
        assert NotImplementedError("Loss not defined")

    def get_acc(self, y, pred):
        assert NotImplementedError("Accuracy not defined")

    def fit(self, X, y):
        X, y = self.fix_inputs(X, y)
        assert len(X.shape) == 2, f"X should have 2 dimensions, but has {len(X.shape)}"
        # Create model
        if len(y.shape) == 1:
            output_dim = y.amax() + 1
        elif len(y.shape) == 2:
            output_dim = y.shape[1]
        else:
            raise ValueError("y not supported")
        self.construct_model(X.shape[1], output_dim)
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
        dataset = TensorDataset(X, y)
        train_size = int(len(dataset) * 0.9) # Use 10% of training data to validate
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,)
        # Start training and logging
        for epoch in range(1, self.max_iter + 1):
            # Train model on training set
            self.model.train()  # set the model to training mode
            total_loss = 0
            for bX, by in train_loader:
                bX, by = bX.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(bX)
                loss = self.get_loss(by, pred)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            # Evaluate model on validation set
            self.model.eval()
            val_acc = 0
            val_prec = 0
            val_rec = 0

            with torch.no_grad():
                for bX, by in val_loader:
                    bX, by = bX.to(self.device), by.to(self.device)
                    pred = self.model(bX)
                    # assert pred.shape == by.shape
                    v_acc, v_prec, v_rec = self.get_acc(by, pred)
                    val_acc += v_acc * bX.shape[0]
                    val_prec += v_prec * bX.shape[0]
                    val_rec += v_rec * bX.shape[0]

            val_acc = val_acc / len(val_dataset)
            val_prec = val_prec / len(val_dataset)
            val_rec = val_rec / len(val_dataset)

            if self.verbose and epoch % 5 == 0 or self.verbose and epoch == 0:
                print(f"Epoch {epoch} - Training Loss: {total_loss:.4f} - Val. Acc.: {val_acc:.2f} - Val. Prec.: {val_prec:.2f} - Val. Rec.: {val_rec:.2f} ")

    def score(self, X, y):
        assert self.model is not None, "Model has not been trained"
        X, y = self.fix_inputs(X, y)
        assert len(X.shape) == 2, f"X should have 2 dimensions, but has {len(X.shape)}"
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        acc = 0
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for bX, by in loader:
                bX, by = bX.to(self.device), by.to(self.device)
                pred = self.model(bX)
                acc += self.get_acc(by, pred)[0] * bX.shape[0]
        return acc.item() / len(dataset)

    def predict(self, X):
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
                pred = self.model(bX)
                outs.append(pred.detach().cpu())
        return torch.cat(outs, dim=0)


class ClsProbe(Probe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loss(self, y, pred):
        return torch.nn.functional.cross_entropy(
            input=pred,
            target=y
        )

    def get_acc(self, y, pred):
        y_pred = pred.argmax(-1).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        accuracy = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        return accuracy, precision, recall

    def fit(self, X, y):
        assert len(y.shape) == 1, f"y should have 1 dimension, but has {len(y.shape)}"
        super().fit(X, y)

    def score(self, X, y):
        assert len(y.shape) == 1, f"y should have 1 dimension, but has {len(y.shape)}"
        return super().score(X, y)

    def predict(self, X):
        out = super().predict(X)
        return out.softmax(dim=-1)


class LinearClsProbe(ClsProbe):

    def __init__(self, fit_intercept: bool = True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_intercept = fit_intercept
        
    def construct_model(self, input_dim, output_dim):
        self.model = nn.Linear(input_dim, output_dim, bias=self.fit_intercept)


class NonlinearClsProbe(ClsProbe):

    def __init__(self, hidden_layer_sizes=(2048,), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        assert len(self.hidden_layer_sizes) >= 1, "Not enough layers"

    def construct_model(self, input_dim, output_dim):
        layers = [nn.Linear(input_dim, self.hidden_layer_sizes[0]), nn.ReLU()] 
        if len(self.hidden_layer_sizes) > 1:
            for prev_size, next_size in zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:]):
                layers += [nn.Linear(prev_size, next_size), nn.ReLU()]
        layers += [nn.Linear(self.hidden_layer_sizes[-1], output_dim)]
        self.model = nn.Sequential(*layers)


class MultiClsProbe(Probe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loss(self, y, pred):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            input=pred,
            target=y,
            pos_weight=torch.tensor([1.0]).to('cuda')
        )

    def get_acc(self, y, pred):
        y_pred = (pred.sigmoid() > 0.5).flatten().detach().cpu().numpy()
        y_true = y.flatten().detach().cpu().numpy()
        accuracy = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        return accuracy, precision, recall

    def fit(self, X, y):
        assert len(y.shape) == 2, f"y should have 2 dimension, but has {len(y.shape)}"
        super().fit(X, y)

    def score(self, X, y):
        assert len(y.shape) == 2, f"y should have 2 dimension, but has {len(y.shape)}"
        return super().score(X, y)

    def predict(self, X):
        out = super().predict(X)
        return out.sigmoid()


class LinearMultiClsProbe(MultiClsProbe):

    def __init__(self, fit_intercept: bool = True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_intercept = fit_intercept

    def construct_model(self, input_dim, output_dim):
        self.model = nn.Linear(input_dim, output_dim, bias=self.fit_intercept)


class NonlinearMultiClsProbe(MultiClsProbe):

    def __init__(self, hidden_layer_sizes=(2048,), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        assert len(self.hidden_layer_sizes) >= 1, "Not enough layers"

    def construct_model(self, input_dim, output_dim):
        layers = [nn.Linear(input_dim, self.hidden_layer_sizes[0]), nn.ReLU()] 
        if len(self.hidden_layer_sizes) > 1:
            for prev_size, next_size in zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:]):
                layers += [nn.Linear(prev_size, next_size), nn.ReLU()]
        layers += [nn.Linear(self.hidden_layer_sizes[-1], output_dim)]
        self.model = nn.Sequential(*layers)


class ConceptErasure:
    
    def __init__(self, X, y):
        self.fit(X, y)
    
    def fit(self, X, y):
        X, y = self.fix_inputs(X, y)
        self.transform = self.leace(X, y)
        self.mean = X.mean(0)

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
        
    def cross_covariance(self, X, Z):
        X_mean = torch.mean(X, 0)
        Z_mean = torch.mean(Z, 0)
        n = X.size(0)
        return (1 / (n - 1)) * (X - X_mean).T @ (Z - Z_mean) 

    def whiten_with_cov(self, cov):
        e, v = torch.linalg.eigh(cov)
        epsilon = 1e-5
        e = torch.sqrt(e + epsilon)
        e = 1. / e
        whitening_matrix = v @ torch.diag(e) @ v.T
        return whitening_matrix

    def leace(self, X, y):
        if len(y.shape) == 1:
            num_classes = y.max().item() + 1  # adding 1 because classes are zero-indexed
            Z = torch.nn.functional.one_hot(y, num_classes).to(torch.float32)
        elif len(y.shape) == 2:
            Z = y.to(torch.float32)
        else:
            raise Exception("Invalid y shape")
        SigmaXX = self.cross_covariance(X, X)
        W = self.whiten_with_cov(SigmaXX)
        SigmaXZ = self.cross_covariance(X, Z)
        prod = W @ SigmaXZ
        P = prod @ torch.linalg.pinv(prod)
        return W @ P @ torch.linalg.pinv(W)
    
    def predict(self, X):
        X = self.fix_inputs(X)
        return X - (X - self.mean) @ self.transform