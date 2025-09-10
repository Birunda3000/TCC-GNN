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

class MusaeGithubLoader(BaseDatasetLoader):
    """Carrega o dataset Musae-Github a partir de arquivos locais."""
    def load(self) -> Tuple[Data, int, List[str]]:
        print("Usando MusaeGithubLoader para carregar dados de arquivos locais...")
        
        df_edges = pd.read_csv(Config.MUSAE_EDGES_PATH)
        edge_index = torch.tensor(df_edges.values.T, dtype=torch.long)
        
        df_target = pd.read_csv(Config.MUSAE_TARGET_PATH)
        df_target = df_target.sort_values(by='id').reset_index(drop=True)
        labels = torch.tensor(df_target['ml_target'].values, dtype=torch.long)
        num_classes = len(df_target['ml_target'].unique())
        
        # Como não temos nomes, criamos nomes genéricos para a legenda
        class_names = [f"Classe {i}" for i in range(num_classes)]
        
        num_nodes = max(df_target['id'].max(), edge_index.max()) + 1
        features = None
        
        graph_data = Data(x=features, edge_index=edge_index, y=labels, num_nodes=num_nodes)
        
        print("Musae-Github carregado com sucesso dos arquivos locais.")
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