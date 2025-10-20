import torch
import os
import random
import numpy as np
import json

from src.config import Config
from src.data_loader import get_loader
from src.classifiers import GCNClassifier, GATClassifier
from src.directory_manager import DirectoryManager


def save_classification_report(run_path, dataset_name, results, reports):
    """Salva um relatório consolidado em formato JSON."""
    summary = {
        "dataset": dataset_name,
        "classification_results": results,
        "detailed_reports": reports,
    }
    report_path = os.path.join(run_path, "graph_classification_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nRelatório de classificação de grafo salvo em: '{report_path}'")


def print_summary_table(results, dataset_name):
    """Imprime a tabela de resumo dos resultados no console."""
    print("\n" + "=" * 65)
    print("RELATÓRIO DE CLASSIFICAÇÃO DE GRAFO (FIM-A-FIM)".center(65))
    print("-" * 65)
    print(f"Dataset: {dataset_name}")
    print("-" * 65)
    print(f"{'Modelo':<25} | {'Acurácia':<12} | {'F1-Score':<12} | {'Tempo (s)':<10}")
    print("=" * 65)
    for name, metrics in results.items():
        print(
            f"{name:<25} | {metrics['accuracy']:<12.4f} | {metrics['f1_score_weighted']:<12.4f} | {metrics['training_time_seconds']:<10.2f}"
        )
    print("=" * 65)


def main():
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE GRAFO (FIM-A-FIM)")
    print(f"Dataset: {config.DATASET_NAME}\n", "=" * 65)

    # --- 2. Carregar Dados e Preparar Ambiente ---
    # **DIFERENÇA CHAVE**: Carrega o dataset original, não um embedding.
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP, run_folder_name="GRAPH_CLASSIFICATION_RUNS"
    )
    run_path = directory_manager.get_run_path()

    results = {}
    reports = {}

    # --- 3. Definir Dimensões e Instanciar Modelos ---
    # A dimensão de entrada é o número total de features do dataset original
    input_dim = wsg_obj.metadata.num_total_features
    # A dimensão de saída é o número de classes únicas
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    # **DIFERENÇA CHAVE**: Instancia modelos GNN que usam a estrutura do grafo.
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

    # --- 5. Salvar e Exibir Resultados ---
    save_classification_report(run_path, config.DATASET_NAME, results, reports)
    print_summary_table(results, config.DATASET_NAME)

    # --- 6. Finalizar Nome do Diretório ---
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_acc = best_model[1]["accuracy"]

    final_path = directory_manager.finalize_run_directory(
        dataset_name=config.DATASET_NAME,
        metrics={"best_acc": best_acc},
    )
    print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")


if __name__ == "__main__":
    main()
