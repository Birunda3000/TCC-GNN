import torch
import random
import numpy as np

from src.config import Config
from src.data_loader import get_loader
from src.classifiers import GCNClassifier, GATClassifier
from src.runner import ExperimentRunner  # <-- Importa a nova classe


def main():
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE GRAFO (FIM-A-FIM)")
    print(f"Dataset de entrada: {config.DATASET_NAME}\n", "=" * 65)

    # --- 2. Carregar Dados ---
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()

    # --- 3. Definir Modelos ---
    input_dim = wsg_obj.metadata.num_total_features
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))
    models_to_run = [
        GCNClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),
        GATClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim, heads=8
        ),
    ]

    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="GRAPH_CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=config.DATASET_NAME,
    )
    runner.run(models_to_run)


if __name__ == "__main__":
    main()
