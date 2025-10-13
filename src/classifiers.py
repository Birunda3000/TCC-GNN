import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from abc import ABC, abstractmethod
from tqdm import tqdm

from torch_geometric.nn import GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .data_format_definition import WSG
from .config import Config
from .data_converter import DataConverter


class BaseClassifier(ABC):
    """
    Classe base abstrata. Define que todo classificador deve saber
    como treinar e se avaliar a partir de um objeto WSG.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.__class__.__name__

    @abstractmethod
    def train_and_evaluate(self, wsg_obj: WSG):
        """
        Orquestra o processo de treinamento e avaliação para o modelo.
        Deve retornar: (acurácia, f1_score, tempo_de_treino, relatório_detalhado)
        """
        pass


class SklearnClassifier(BaseClassifier):
    """Wrapper para modelos Scikit-learn que contém sua própria lógica de treino."""

    def __init__(self, config: Config, model_class, **model_params):
        super().__init__(config)
        self.model_name = model_class.__name__
        try:
            self.model = model_class(random_state=config.RANDOM_SEED, **model_params)
        except TypeError:
            self.model = model_class(**model_params)

    def train_and_evaluate(self, wsg_obj: WSG):
        print(f"\n--- Avaliando (Sklearn): {self.model_name} ---")

        pyg_data = DataConverter.to_pyg_data(wsg_obj)
        X = pyg_data.x.cpu().numpy()
        y = pyg_data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto pyg_data
        X_train, y_train = X[pyg_data.train_mask], y[pyg_data.train_mask]
        X_test, y_test = X[pyg_data.test_mask], y[pyg_data.test_mask]

        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        return acc, f1, train_time, report


class PyTorchClassifier(BaseClassifier, nn.Module):
    """
    Classe base para classificadores PyTorch. Contém o loop de treino completo.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        BaseClassifier.__init__(self, config)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _train_step(self, optimizer, criterion, data, use_gnn):
        self.train()
        optimizer.zero_grad()

        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test_step(self, data, use_gnn):
        self.eval()
        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]

        acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average="weighted")
        report = classification_report(
            y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
        )

        return acc, f1, report

    def _train_and_evaluate_internal(self, wsg_obj: WSG, use_gnn: bool):
        print(f"\n--- Avaliando (PyTorch): {self.model_name} ---")
        device = torch.device(self.config.DEVICE)
        self.to(device)

        data = DataConverter.to_pyg_data(wsg_obj).to(device)
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        pbar = tqdm(
            range(self.config.EPOCHS),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )
        for epoch in pbar:
            loss = self._train_step(optimizer, criterion, data, use_gnn)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        train_time = time.time() - start_time

        acc, f1, report = self._test_step(data, use_gnn)
        return acc, f1, train_time, report


# --- Implementações Específicas ---


class MLPClassifier(PyTorchClassifier):
    """Classificador MLP que opera em um tensor de features denso."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_and_evaluate(self, wsg_obj: WSG):
        return self._train_and_evaluate_internal(wsg_obj, use_gnn=False)


class GCNClassifier(PyTorchClassifier):
    """Classificador GCN que opera em features e na estrutura do grafo."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

    def train_and_evaluate(self, wsg_obj: WSG):
        return self._train_and_evaluate_internal(wsg_obj, use_gnn=True)


class GATClassifier(PyTorchClassifier):
    """Classificador GAT que utiliza mecanismos de atenção."""

    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=8):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)

    def train_and_evaluate(self, wsg_obj: WSG):
        return self._train_and_evaluate_internal(wsg_obj, use_gnn=True)
