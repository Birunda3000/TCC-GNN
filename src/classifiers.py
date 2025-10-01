import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# --- Modelo Categoria 1: Agnóstico à Estrutura do Grafo ---


class MLPClassifier(nn.Module):
    """
    Classificador MLP que opera apenas nas features dos nós.
    Pode lidar com features esparsas (via EmbeddingBag) ou densas.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, sparse=False, vocab_size=None
    ):
        super().__init__()
        self.sparse = sparse

        if self.sparse:
            # Se as features são esparsas, usamos EmbeddingBag para criar uma representação densa
            self.embedder = nn.EmbeddingBag(vocab_size, input_dim, mode="sum")

        # Camadas lineares do MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        if self.sparse:
            # Processa features esparsas para obter um tensor denso
            x = self.embedder(
                data.feature_indices, data.feature_offsets, data.feature_weights
            )
        else:
            # Usa features densas (embeddings) diretamente
            x = data.x

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Modelos Categoria 2: Dependentes da Estrutura do Grafo ---


class GCNClassifier(nn.Module):
    """
    Classificador GCN que utiliza tanto as features dos nós quanto a estrutura do grafo.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GATClassifier(nn.Module):
    """
    Classificador GAT que utiliza mecanismos de atenção na estrutura do grafo.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super().__init__()
        # A primeira camada GAT transforma as features de entrada em 'hidden_dim' usando 'heads' de atenção
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        # A segunda camada GAT transforma as features concatenadas das heads para a dimensão de saída final
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x
