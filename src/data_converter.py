# src/data_converter.py

import torch
from torch_geometric.data import Data

from src.data_format_definition import WSG


class DataConverter:
    """
    Converte um objeto WSG validado para um objeto torch_geometric.data.Data,
    pronto para ser usado em modelos GNN.
    """

    @staticmethod
    def to_pyg_data(wsg_obj: WSG) -> Data:
        """
        Converte um objeto WSG para um objeto torch_geometric.data.Data.
        """
        print("Convertendo objeto WSG para formato PyTorch Geometric...")

        # Extrai o edge_index - CORREÇÃO: Lida com diferentes formatos possíveis
        edge_data = wsg_obj.graph_structure.edge_index
        if isinstance(edge_data, list):
            # Verifica o formato real dos dados
            if len(edge_data) > 0:
                first_item = edge_data[0]
                if isinstance(first_item, list) and len(first_item) == 2:
                    # Formato: [[src, dst], [src, dst], ...]
                    edge_index = torch.tensor(edge_data, dtype=torch.long).t()
                elif isinstance(first_item, int):
                    # Formato: [src, dst, src, dst, ...] (lista plana)
                    edge_index = (
                        torch.tensor(edge_data, dtype=torch.long).view(-1, 2).t()
                    )
                else:
                    # Tenta converter assumindo uma lista de pares
                    try:
                        edges = [[int(e[0]), int(e[1])] for e in edge_data]
                        edge_index = torch.tensor(edges, dtype=torch.long).t()
                    except (IndexError, TypeError):
                        print(
                            f"WARNING: Formato inesperado do edge_index. Primeira entrada: {first_item}"
                        )
                        # Fallback para um grafo vazio
                        edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                # Lista vazia
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            # Tenta tratar como um tipo iterável genérico
            try:
                edges = [[int(src), int(dst)] for src, dst in edge_data]
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            except Exception as e:
                print(f"ERROR: Não foi possível converter edge_index: {e}")
                edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Processa as features dos nós
        num_nodes = wsg_obj.metadata.num_nodes
        feature_type = wsg_obj.metadata.feature_type

        if feature_type == "sparse_binary":
            # Para features esparsas, usa índices para criar um tensor one-hot
            num_features = wsg_obj.metadata.num_total_features
            x = torch.zeros((num_nodes, num_features), dtype=torch.float)
            for node_id, feature in wsg_obj.node_features.items():
                indices = feature.indices
                node_idx = int(node_id)
                x[node_idx, indices] = 1.0
        else:  # dense_continuous
            # Para features densas (embeddings), diretamente usa os pesos
            feature_dim = len(next(iter(wsg_obj.node_features.values())).weights)
            x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
            for node_id, feature in wsg_obj.node_features.items():
                node_idx = int(node_id)
                x[node_idx] = torch.tensor(feature.weights, dtype=torch.float)

        # Extrai os rótulos (y)
        y_list = wsg_obj.graph_structure.y

        # Lida com possíveis valores None nos rótulos
        valid_indices = [i for i, y_val in enumerate(y_list) if y_val is not None]
        valid_y = [y_list[i] for i in valid_indices]

        if not valid_y:
            raise ValueError("Nenhum rótulo válido encontrado no objeto WSG.")

        y = torch.tensor([int(label) for label in valid_y], dtype=torch.long)

        # Cria máscaras para treino/teste
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            valid_indices, train_size=0.8, random_state=42
        )

        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(
            x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask
        )

        print("Conversão concluída com sucesso.")
        return data
