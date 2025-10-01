import torch
import torch.optim as optim
import os
import time
import random
import numpy as np

from src.config import Config
from src.data_loader import get_loader, DirectWSGLoader
from src.data_converter import DataConverter
import src.classifiers as classifiers
from src.train import train_model, save_results, save_report
from src.directory_manager import DirectoryManager

def main():
    """
    Main function to run feature classification experiments.
    """
    config = Config()
    device = torch.device(config.DEVICE)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    loader = DirectWSGLoader(file_path="data/wsg_data.csv")
    wsg_obj = loader.load()
    pyg_data = DataConverter.to_pyg_data(wsg_obj).to(device)


    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name="CLASSIFICATION_RUNS"
    )

    models = {
        "LogisticRegression": classifiers.LogisticRegression,
        "SVM": classifiers.SVM,
        "RandomForest": classifiers.RandomForest,
        "MLP": classifiers.MLP
    }

