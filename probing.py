import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                    val_acc += self.get_acc(by, pred) * bX.shape[0]

                    pred_binary = (pred > 0.5).type_as(by)
                    val_prec += precision_score(by.cpu().flatten(), pred_binary.cpu().flatten()) * bX.shape[0]
                    val_rec += recall_score(by.cpu().flatten(), pred_binary.cpu().flatten()) * bX.shape[0]
            val_acc = val_acc / len(val_dataset)
            val_prec = val_prec / len(val_dataset)
            val_rec = val_rec / len(val_dataset)
            if self.verbose and epoch % 20 == 0 or self.verbose and epoch == 0:
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
                acc += self.get_acc(by, pred) * bX.shape[0]
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
        top_pred = pred.argmax(-1)
        return torch.sum(y == top_pred) / pred.shape[0]

    def fit(self, X, y):
        assert len(y.shape) == 1, f"y should have 1 dimension, but has {len(y.shape)}"
        super().fit(X, y)

    def score(self, X, y):
        assert len(y.shape) == 1, f"y should have 1 dimension, but has {len(y.shape)}"
        return super().score(X, y)

    def predict(self, X):
        out = super().predict(X)
        return out.argmax(-1)


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
            #pos_weight=torch.tensor([2.5]).to('cuda')
        )

    def get_acc(self, y, pred):
        pred = pred.sigmoid()
        top_pred = (pred > 0.5) == y
        return torch.sum(top_pred) / (pred.shape[0] * pred.shape[1])

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


class RegressionProbe(Probe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loss(self, y, pred):
        return torch.nn.functional.mse_loss(
            input=pred,
            target=y
        )

    def get_acc(self, y, pred):
        top_pred = (pred > 0.5) == y
        return torch.sum(top_pred) / (pred.shape[0] * pred.shape[1])

    def fit(self, X, y):
        assert len(y.shape) == 2, f"y should have 2 dimension, but has {len(y.shape)}"
        super().fit(X, y)

    def score(self, X, y):
        assert len(y.shape) == 2, f"y should have 2 dimension, but has {len(y.shape)}"
        return super().score(X, y)

    def predict(self, X):
        out = super().predict(X)
        return out


class LinearRegressionProbe(RegressionProbe):

    def __init__(self, fit_intercept: bool = True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_intercept = fit_intercept
        
    def construct_model(self, input_dim, output_dim):
        self.model = nn.Linear(input_dim, output_dim, bias=self.fit_intercept)


class NonlinearRegressionProbe(RegressionProbe):

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
