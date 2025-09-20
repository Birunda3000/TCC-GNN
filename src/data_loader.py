# src/data_loader.py

import torch
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from typing import Tuple, List
import pandas as pd
import numpy as np
import json

from config import Config

class BaseDatasetLoader(ABC):
    """Classe base que define o contrato para os loaders."""
    @abstractmethod
    def load(self) -> Tuple[Data, int, List[str]]:
        """
        Método de carregamento principal.
        Retorna:
            Tuple[Data, int, List[str]]: Uma tupla contendo o objeto do grafo,
                                         o número de classes e uma lista com os nomes das classes.
        """
        pass

class CoraLoader(BaseDatasetLoader):
    """Carrega o dataset Cora a partir de arquivos locais."""
    def load(self) -> Tuple[Data, int, List[str]]:
        print("Usando CoraLoader para carregar dados de arquivos locais...")
        
        df_content = pd.read_csv(Config.CORA_CONTENT_PATH, sep='\t', header=None, dtype={0: str})
        
        paper_ids = df_content.iloc[:, 0].tolist()
        paper_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}
        
        features = torch.tensor(df_content.iloc[:, 1:-1].values, dtype=torch.float)
        
        # Agora capturamos os nomes das classes do pandas
        labels, class_names = pd.factorize(df_content.iloc[:, -1])
        labels = torch.tensor(labels, dtype=torch.long)
        
        df_cites = pd.read_csv(Config.CORA_CITES_PATH, sep='\t', header=None, dtype=str)
        
        valid_edges = df_cites[df_cites[0].isin(paper_to_idx) & df_cites[1].isin(paper_to_idx)]
        src = [paper_to_idx[paper] for paper in valid_edges[0]]
        dst = [paper_to_idx[paper] for paper in valid_edges[1]]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        graph_data = Data(x=features, edge_index=edge_index, y=labels)
        
        print("Cora carregado com sucesso dos arquivos locais.")
        # Retornamos os nomes das classes como uma lista de strings
        return graph_data, len(class_names), class_names.tolist()

# Em src/data_loader.py
import itertools

class MusaeGithubLoader(BaseDatasetLoader):
    """
    Carrega o dataset Musae-Github, tratando suas features esparsas
    para uso com uma camada EmbeddingBag.
    """
    def load(self) -> Tuple[Data, int, List[str]]:
        print("Usando MusaeGithubLoader para carregar dados (modo features esparsas)...")
        
        # Carregamento de arestas e labels (permanece igual)
        df_edges = pd.read_csv(Config.MUSAE_EDGES_PATH)
        edge_index = torch.tensor(df_edges.values.T, dtype=torch.long)
        
        df_target = pd.read_csv(Config.MUSAE_TARGET_PATH)
        df_target = df_target.sort_values(by='id').reset_index(drop=True)
        labels = torch.tensor(df_target['ml_target'].values, dtype=torch.long)
        num_classes = len(df_target['ml_target'].unique())
        class_names = [f"Classe {i}" for i in range(num_classes)]
        
        num_nodes = int(df_target['id'].max()) + 1
        
        # --- NOVO TRATAMENTO DAS FEATURES ESPARSAS ---
        with open(Config.MUSAE_FEATURES_PATH, 'r') as f:
            feature_data = json.load(f)
        
        # Ordena os dados de features pelo ID do nó (de 0 a num_nodes-1)
        # Isso garante que a ordem dos offsets corresponda à ordem dos nós
        sorted_features = [feature_data[str(i)] for i in range(num_nodes)]

        # 1. Cria o tensor 1D gigante com todos os índices
        # A função chain.from_iterable é uma forma eficiente de achatar a lista de listas
        all_feature_indices = list(itertools.chain.from_iterable(sorted_features))
        feature_indices = torch.tensor(all_feature_indices, dtype=torch.long)

        # 2. Cria os offsets
        # O primeiro offset é sempre 0
        offsets = [0] + list(itertools.accumulate(len(f) for f in sorted_features))[:-1]
        feature_offsets = torch.tensor(offsets, dtype=torch.long)
        
        # 3. Descobre o número total de features únicas
        num_total_features = int(feature_indices.max().item() + 1)
        print(f"Total de features únicas (tamanho do dicionário): {num_total_features}")

        graph_data = Data(
            edge_index=edge_index, 
            y=labels, 
            num_nodes=num_nodes,
            # Adicionamos os novos atributos ao objeto Data
            feature_indices=feature_indices,
            feature_offsets=feature_offsets
        )
        
        # Adicionamos metadados que serão úteis para construir o modelo
        graph_data.num_total_features = num_total_features
        
        print("Musae-Github (esparso) carregado com sucesso.")
        return graph_data, num_classes, class_names

def get_loader(dataset_name: str) -> BaseDatasetLoader:
    """Função Fábrica que retorna uma instância do loader correto."""
    if dataset_name.lower() == 'cora':
        return CoraLoader()
    elif dataset_name.lower() == 'musae-github':
        return MusaeGithubLoader()
    elif dataset_name.lower() == 'musae-facebook':
        raise NotImplementedError("Loader para Musae-Facebook ainda não implementado.")
    else:
        raise ValueError(f"Loader para o dataset '{dataset_name}' não foi implementado.")