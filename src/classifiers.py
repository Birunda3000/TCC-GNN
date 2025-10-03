import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# --- Categoria 1: Modelos Agnósticos à Estrutura do Grafo ---


class MLPClassifier(nn.Module):
    """Classificador MLP que opera em um tensor de features denso."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- Categoria 2: Modelos que Utilizam a Estrutura do Grafo ---


class GCNClassifier(nn.Module):
    """Classificador GCN que opera em features e na estrutura do grafo."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GATClassifier(nn.Module):
    """Classificador GAT que utiliza mecanismos de atenção."""

    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)
