# src/data_converter.py

import torch
import itertools
from torch_geometric.data import Data

from src.data_format_definition import WSG

class DataConverter:
    """
    Converte um objeto WSG validado para um objeto torch_geometric.data.Data,
    pronto para ser usado em modelos GNN.
    """

    @staticmethod
    def to_pyg_data(wsg_data: WSG) -> Data:
        """
        Realiza a conversão de WSG para o formato de dados do PyTorch Geometric.

        Este método:
        1. Converte a estrutura do grafo (edge_index, y) para tensores.
        2. Processa o dicionário `node_features` para criar os tensores
           `feature_indices`, `feature_offsets` e `feature_weights`
           necessários para a camada `nn.EmbeddingBag`.
        3. Anexa metadados importantes ao objeto Data final.

        Args:
            wsg_data (WSG): O objeto de dados validado pelo Pydantic.

        Returns:
            Data: Um objeto torch_geometric.data.Data pronto para treinamento.
        """
        print("Convertendo objeto WSG para formato PyTorch Geometric...")

        # --- 1. Converter Estrutura do Grafo ---
        edge_index = torch.tensor(wsg_data.graph_structure.edge_index, dtype=torch.long)

        # Converte labels, tratando valores None (comuns em nós de teste/validação)
        # Usamos -1 como um valor padrão para labels ausentes.
        y = torch.tensor(
            [-1 if label is None else label for label in wsg_data.graph_structure.y],
            dtype=torch.long,
        )

        # --- 2. Processar Features para EmbeddingBag ---
        num_nodes = wsg_data.metadata.num_nodes
        
        # Listas para agregar dados de todos os nós
        all_indices = []
        all_weights = []
        offsets = [0]  # O primeiro offset é sempre 0

        # Itera de 0 a N-1 para garantir a ordem correta
        for i in range(num_nodes):
            node_id_str = str(i)
            node_feat = wsg_data.node_features[node_id_str]
            
            all_indices.extend(node_feat.indices)
            all_weights.extend(node_feat.weights)
            # O próximo offset é o offset atual + o número de features deste nó
            offsets.append(offsets[-1] + len(node_feat.indices))
        
        # Remove o último offset, que é apenas o comprimento total
        offsets.pop()

        feature_indices = torch.tensor(all_indices, dtype=torch.long)
        feature_weights = torch.tensor(all_weights, dtype=torch.float)
        feature_offsets = torch.tensor(offsets, dtype=torch.long)

        # --- 3. Montar o Objeto Data ---
        pyg_data = Data(
            edge_index=edge_index,
            y=y,
            feature_indices=feature_indices,
            feature_offsets=feature_offsets,
            feature_weights=feature_weights,
            num_nodes=num_nodes,
        )
        
        # Anexamos metadados importantes para fácil acesso durante a instanciação do modelo
        pyg_data.num_total_features = wsg_data.metadata.num_total_features
        pyg_data.dataset_name = wsg_data.metadata.dataset_name

        print("Conversão concluída com sucesso.")
        return pyg_data