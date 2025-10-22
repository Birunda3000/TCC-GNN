import torch
import os
import random
import numpy as np
import json

from src.config import Config
from src.data_loader import get_loader
from src.classifiers import GCNClassifier, GATClassifier
from src.directory_manager import DirectoryManager
from src.data_converter import DataConverter


def main():
    # --- 1. Configuração Inicial ---
    config = Config()
    # device = torch.device(config.DEVICE) # O dispositivo é gerenciado dentro de cada classificador

    # Aplica a semente de aleatoriedade para reprodutibilidade
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE GRAFO (FIM-A-FIM)")
    print(f"Dataset de entrada: {config.DATASET_NAME}\n", "=" * 65)

    # --- 2. Carregar Dados e Preparar Ambiente ---
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()
    # A conversão para PyG é feita dentro de cada classificador GNN

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP, run_folder_name="GRAPH_CLASSIFICATION_RUNS"
    )
    # run_path não é mais necessário aqui, pois é gerenciado internamente

    results = {}
    reports = {}

    # --- 3. Definir Dimensões e Instanciar Modelos ---
    # A dimensão de entrada é o número total de features do dataset original
    input_dim = wsg_obj.metadata.num_total_features
    # A dimensão de saída é o número de classes únicas
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    models_to_run = [
        GCNClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),
        GATClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim, heads=8
        ),
    ]

    # --- 4. Iterar, Treinar e Avaliar cada modelo ---
    for model in models_to_run:
        acc, f1, train_time, report = model.train_and_evaluate(wsg_obj)

        results[model.model_name] = {
            "accuracy": acc,
            "f1_score_weighted": f1,
            "training_time_seconds": train_time,
        }
        if report:
            reports[f"{model.model_name}_classification_report"] = report

    # --- 5. Salvar e Exibir Resultados usando DirectoryManager ---
    directory_manager.save_classification_report(
        input_file=config.DATASET_NAME, results=results, reports=reports
    )
    directory_manager.print_summary_table(
        results=results,
        input_file_path=config.DATASET_NAME,
        feature_type=wsg_obj.metadata.feature_type,
    )

    # --- 6. Finalizar Nome do Diretório ---
    if results:
        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        best_acc = best_model[1]["accuracy"]
        best_model_name = best_model[0].lower().replace("classifier", "")

        final_path = directory_manager.finalize_run_directory(
            dataset_name=wsg_obj.metadata.dataset_name,
            metrics={"best_acc": best_acc, "model": best_model_name},
        )
        print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")
    else:
        print("\nNenhum resultado para finalizar o diretório.")


if __name__ == "__main__":
    main()
