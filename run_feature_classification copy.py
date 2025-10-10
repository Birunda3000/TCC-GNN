import torch
import torch.optim as optim
import os
import random
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import Config
from src.data_loader import DirectWSGLoader
from src.classifiers import MLPClassifier, GCNClassifier, GATClassifier
from src.train_loop import train_and_evaluate_sklearn_model, run_pytorch_classification
from src.directory_manager import DirectoryManager
from src.data_loader import get_loader                                                                                                                  



def main():
        # --- 1. Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)

    # Aplica a semente de aleatoriedade para reprodutibilidade
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name="CLASSIFICATION_RUNS"
    )
    run_path = directory_manager.get_run_path()

    models = [
        MLPClassifier,
        GCNClassifier,
        GATClassifier,
        LogisticRegression,
        RandomForestClassifier
    ]
