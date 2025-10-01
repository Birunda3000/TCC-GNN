# src/data_loader.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
from src.config import Config
from src.data_format_definition import WSG, Metadata, GraphStructure, NodeFeaturesEntry


class BaseDatasetLoader(ABC):
    """Classe base que define o contrato para os loaders de dataset."""

    @abstractmethod
    def load(self) -> WSG:
        """
        Método de carregamento principal.

        Deve carregar os dados brutos de um dataset e transformá-los em um
        objeto que segue a especificação do formato Weighted Sparse Graph (WSG).

        Returns:
            WSG: Um objeto Pydantic representando o grafo no formato WSG.
        """
        pass

class DirectWSGLoader(BaseDatasetLoader):
    """Carrega um dataset já no formato WSG a partir de um arquivo JSON local."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> WSG:
        """
        Carrega o arquivo JSON e o valida como um objeto WSG.

        Returns:
            WSG: Um objeto Pydantic contendo o grafo completo e validado no formato WSG.
        """
        with open(self.file_path, "r") as f:
            wsg_data: Dict[str, Any] = json.load(f)

        # A instanciação do modelo Pydantic valida automaticamente a estrutura e os tipos.
        wsg_object = WSG(**wsg_data)
        return wsg_object

class CoraLoader(BaseDatasetLoader):
    """Carrega o dataset Cora a partir de arquivos locais."""

    def load(self) -> WSG:
        """
        Carrega e processa o dataset Cora para o formato WSG.

        Raises:
            NotImplementedError: Esta função ainda não foi implementada.
        """
        # TODO: Implementar a lógica de carregamento e processamento para o dataset Cora.
        raise NotImplementedError(
            "O loader para o dataset Cora ainda não foi implementado."
        )


# Em src/data_loader.py
import itertools

class MusaeGithubLoader(BaseDatasetLoader):
    """Carrega o dataset Musae-Github a partir de arquivos locais."""

    def load(self) -> WSG:
        """
        Carrega os dados brutos do Musae-Github e os transforma para o formato WSG.

        O processo consiste em:
        1. Carregar as arestas, alvos (labels) e features dos arquivos CSV e JSON.
        2. Construir os dicionários para metadados, estrutura do grafo e features.
        3. Instanciar o objeto Pydantic `WSG`, que valida automaticamente a estrutura e os tipos.
        4. Retornar o objeto `WSG` validado.

        Returns:
            WSG: Um objeto Pydantic contendo o grafo completo e validado no formato WSG.
        """
        edges_df = pd.read_csv(Config.GITHUB_MUSAE_EDGES_PATH)
        target_df = pd.read_csv(Config.GITHUB_MUSAE_TARGET_PATH)
        with open(Config.GITHUB_MUSAE_FEATURES_PATH, "r") as f:
            features_json: Dict[str, List[int]] = json.load(f)

        print("Arquivos carregados. Iniciando processamento para o formato WSG...")

        # --- 1. Preparar dados para os modelos Pydantic ---

        # Garante que arestas não direcionadas sejam únicas e bidirecionais
        # Cria pares (min(u,v), max(u,v)) para identificar arestas únicas
        unique_edges = set(
            tuple(sorted(edge)) for edge in edges_df.itertuples(index=False, name=None)
        )

        source_nodes = [u for u, v in unique_edges] + [v for u, v in unique_edges]
        target_nodes = [v for u, v in unique_edges] + [u for u, v in unique_edges]

        num_nodes: int = len(target_df)
        num_edges: int = len(source_nodes)

        all_indices = (idx for indices in features_json.values() for idx in indices)
        try:
            max_feature_index = max(all_indices)
            num_total_features = max_feature_index + 1
        except ValueError:
            num_total_features = 0

        tz_offset = timedelta(hours=-3)
        tz_info = timezone(tz_offset)
        processed_at: str = datetime.now(tz_info).isoformat()

        metadata_data = {
            "dataset_name": "Musae-Github",
            "feature_type": "sparse_binary",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_total_features": num_total_features,
            "processed_at": processed_at,
            "directed": False,
        }

        graph_structure_data = {
            "edge_index": [
                source_nodes,
                target_nodes,
            ],
            "y": target_df["ml_target"]
            .where(pd.notnull(target_df["ml_target"]), None)
            .tolist(),
            "node_names": target_df["name"].tolist(),
        }

        # Garante que todos os nós de 0 a num_nodes-1 tenham uma entrada de feature.
        # Se um nó não estiver em features_json, ele recebe listas vazias.
        node_features_data = {
            str(i): {
                "indices": features_json.get(str(i), []),
                "weights": [1.0] * len(features_json.get(str(i), [])),
            }
            for i in range(num_nodes)
        }

        # --- 2. Instanciar e validar o objeto WSG ---
        # A instanciação dos modelos Pydantic substitui as asserções manuais.
        # Se os dados não estiverem no formato correto, Pydantic levantará um `ValidationError`.
        wsg_object = WSG(
            metadata=Metadata(**metadata_data),
            graph_structure=GraphStructure(**graph_structure_data),
            node_features={
                k: NodeFeaturesEntry(**v) for k, v in node_features_data.items()
            },
        )

        print("Processamento e validação com Pydantic concluídos com sucesso.")
        return wsg_object


def get_loader(dataset_name: str) -> BaseDatasetLoader:
    """Função Fábrica que retorna uma instância do loader correto."""
    if dataset_name.lower() == "cora":
        return CoraLoader()
    elif dataset_name.lower() == "musae-github":
        return MusaeGithubLoader()
    elif dataset_name.lower() == "musae-facebook":
        raise NotImplementedError("Loader para Musae-Facebook ainda não implementado.")
    else:
        raise ValueError(
            f"Loader para o dataset '{dataset_name}' não foi implementado."
        )
