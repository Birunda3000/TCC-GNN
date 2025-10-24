import torch
import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.config import Config
from src.data_loader import DirectWSGLoader
from src.classifiers import SklearnClassifier, MLPClassifier, XGBoostClassifier
from src.runner import ExperimentRunner

try:
    import xgboost

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def main():
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    # --- 2. Carregar Dados ---
    wsg_file_path = "data/output/EMBEDDING_RUNS/musae-github__loss_0_9483__emb_dim_32__01-10-2025_14-58-11/musae-github_embeddings.wsg.json"
    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE EMBEDDINGS")
    print(f"Arquivo de entrada: {wsg_file_path}\n", "=" * 65)
    loader = DirectWSGLoader(file_path=wsg_file_path)
    wsg_obj = loader.load()

    # --- 3. Definir Modelos ---
    input_dim = len(wsg_obj.node_features["0"].weights)
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))
    models_to_run = [
        SklearnClassifier(config, model_class=LogisticRegression, max_iter=1000),
        SklearnClassifier(config, model_class=KNeighborsClassifier, n_neighbors=5),
        SklearnClassifier(config, model_class=RandomForestClassifier),
        MLPClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),
    ]
    if XGBOOST_AVAILABLE:
        models_to_run.append(XGBoostClassifier(config))

    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="FEATURE_CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=os.path.basename(wsg_file_path),
    )
    runner.run(models_to_run)


if __name__ == "__main__":
    main()
