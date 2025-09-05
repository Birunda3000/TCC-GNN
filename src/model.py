import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    Modelo de Graph Convolutional Network (GCN) com duas camadas.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Primeira camada convolucional + ativação ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # Segunda camada convolucional (saída para classificação)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)