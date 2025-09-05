import torch
from typing import Tuple
from torch_geometric.data import Data
import random
import numpy as np


def set_seeds(seed: int):
    """
    Define as sementes para random, numpy e torch para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Sementes aleatórias definidas como: {seed}")


def create_deterministic_masks(
    data: Data, split_ratio: Tuple[float, float, float]
) -> Data:
    """
    Cria máscaras de treino, validação e teste de forma determinística.

    A divisão é feita sequencialmente sobre os nós: os primeiros N% são para treino,
    os M% seguintes para validação, e o restante para teste.

    Args:
        data (Data): O objeto de dados do grafo.
        split_ratio (Tuple[float, float, float]): Uma tupla com as proporções
                                                   para (treino, validação, teste).

    Returns:
        Data: O objeto de dados com as máscaras 'train_mask', 'val_mask', 'test_mask'.
    """
    train_ratio, val_ratio, _ = split_ratio
    num_nodes = data.num_nodes

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    # Cria um tensor de índices de 0 a num_nodes-1
    indices = torch.arange(num_nodes)

    # Define as máscaras como tensores booleanos vazios
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Preenche as máscaras com base nos índices sequenciais
    data.train_mask[indices[:num_train]] = True
    data.val_mask[indices[num_train : num_train + num_val]] = True
    data.test_mask[indices[num_train + num_val :]] = True

    print(f"Divisão de dados determinística criada:")
    print(f"  - Treino: {data.train_mask.sum()} nós ({train_ratio:.0%})")
    print(f"  - Validação: {data.val_mask.sum()} nós ({val_ratio:.0%})")
    print(f"  - Teste: {data.test_mask.sum()} nós")

    return data
