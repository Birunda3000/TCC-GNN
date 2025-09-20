# Em src/model.py
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GCNEncoder(Module):
    """Encoder GCN para o VGAE. Adapta-se a features esparsas ou densas."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Duas "cabeças" no final: uma para a média (mu) e outra para a variância (logstd)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        # Retorna a média e o log da variância para cada nó
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGAE(Module):
    """
    Modelo Variational Graph Autoencoder completo.
    Ele usa o GCNEncoder para gerar embeddings e reconstrói a matriz de adjacência.
    """
    def __init__(self, num_total_features: int, embedding_dim: int, hidden_dim: int, out_embedding_dim: int):
        super().__init__()
        
        # Camada para lidar com as features esparsas de entrada (se necessário)
        # Esta camada transforma os índices esparsos em vetores densos de 'embedding_dim'
        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features, 
            embedding_dim=embedding_dim, 
            mode='mean'
        )
        
        # O encoder GNN que processará os vetores densos
        self.encoder = GCNEncoder(embedding_dim, hidden_dim, out_embedding_dim)
        
        # Guardaremos os valores computados de mu e logstd
        self.__mu__ = self.__logstd__ = None

    def encode(self, data):
        # Gera a matriz de features 'x' a partir dos dados esparsos
        x = self.feature_embedder(data.feature_indices, data.feature_offsets)
        edge_index = data.edge_index
        
        # Usa o encoder para obter os parâmetros da distribuição
        self.__mu__, self.__logstd__ = self.encoder(x, edge_index)
        
        # Amostragem (reparameterization trick) para obter o embedding final Z
        gaussian_noise = torch.randn_like(self.__mu__)
        z = self.__mu__ + gaussian_noise * torch.exp(self.__logstd__)
        return z

    def decode(self, z, edge_index):
        # Decoder simples baseado em produto escalar
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def kl_loss(self):
        # A loss de regularização que organiza o espaço latente
        kl = -0.5 * torch.mean(
            torch.sum(1 + 2 * self.__logstd__ - self.__mu__.pow(2) - self.__logstd__.exp().pow(2), dim=1)
        )
        return kl

    def reconstruction_loss(self, z, pos_edge_index):
        # A loss de reconstrução. O objetivo é maximizar a probabilidade de arestas existentes.
        # Adicionamos também amostragem de arestas negativas para aprender o que NÃO conectar.
        
        # Amostragem de arestas negativas (que não existem no grafo)
        num_nodes = z.size(0)
        neg_edge_index = torch.randint(0, num_nodes, pos_edge_index.size(), dtype=torch.long, device=z.device)

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        
        y = torch.cat([pos_y, neg_y], dim=0)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        
        logits = self.decode(z, edge_index)
        
        return F.binary_cross_entropy_with_logits(logits, y)

    def get_embeddings(self, data):
        # Após o treino, usamos apenas a média (mu) como o embedding final,
        # pois é a representação mais estável.
        with torch.no_grad():
            x = self.feature_embedder(data.feature_indices, data.feature_offsets)
            mu, _ = self.encoder(x, data.edge_index)
        return mu