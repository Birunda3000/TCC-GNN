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
from src.classifiers import MLPClassifier, GCNClassifier
from src.train_loop import train_and_evaluate_sklearn_model, run_pytorch_classification
from src.directory_manager import DirectoryManager

def run_all_classifiers(wsg_obj, config):
    """
    Executa todos os classificadores e retorna um dicionário com os resultados.
    """
    results = {}
    reports = {}

    # --- 1. Modelos Scikit-learn ---
    print("\n--- Avaliando modelos Scikit-learn ---")
    for name, model in {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_SEED
        ),
        "RandomForest": RandomForestClassifier(random_state=config.RANDOM_SEED),
    }.items():
        acc, f1, train_time = train_and_evaluate_sklearn_model(model, wsg_obj, config)
        results[name] = {
            "accuracy": acc,
            "f1_score_weighted": f1,
            "training_time_seconds": train_time,
        }

    # --- 2. Modelo MLP (PyTorch) ---
    print("\n--- Avaliando modelo MLP (PyTorch) ---")
    is_sparse = wsg_obj.metadata.feature_type == "sparse_binary"
    # A dimensão de entrada para features esparsas é o tamanho do vocabulário
    input_dim = (
        wsg_obj.metadata.num_total_features
        if is_sparse
        else wsg_obj.node_features["0"].indices.shape[0]
    )

    # Se estivermos usando embeddings, a dimensão de entrada é a dimensão do embedding
    if not is_sparse:
        # Assumimos que o wsg de embedding tem features densas
        # e o tamanho pode ser pego do primeiro nó.
        # Esta é uma simplificação; o ideal seria ter isso no metadata.
        input_dim = len(wsg_obj.node_features["0"].weights)

    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    mlp_model = MLPClassifier(
        input_dim=input_dim, hidden_dim=128, output_dim=output_dim
    )
    acc, f1, train_time, report = run_pytorch_classification(
        model=mlp_model,
        wsg_obj=wsg_obj,
        config=config,
        optimizer_class=optim.Adam,
        criterion=torch.nn.CrossEntropyLoss(),
        use_gnn=False,
    )
    results["MLP"] = {
        "accuracy": acc,
        "f1_score_weighted": f1,
        "training_time_seconds": train_time,
    }
    reports["MLP_classification_report"] = report

    # --- 3. Modelo GCN (PyTorch) ---
    print("\n--- Avaliando modelo GCN (PyTorch) ---")
    gcn_model = GCNClassifier(
        input_dim=input_dim, hidden_dim=128, output_dim=output_dim
    )
    acc, f1, train_time, report_gcn = run_pytorch_classification(
        model=gcn_model,
        wsg_obj=wsg_obj,
        config=config,
        optimizer_class=optim.Adam,
        criterion=torch.nn.CrossEntropyLoss(),
        use_gnn=True,
    )
    results["GCN"] = {
        "accuracy": acc,
        "f1_score_weighted": f1,
        "training_time_seconds": train_time,
    }
    reports["GCN_classification_report"] = report_gcn

    return results, reports


def save_classification_report(run_path, input_file, results, reports):
    """Salva um relatório consolidado em formato JSON."""
    summary = {
        "input_wsg_file": input_file,
        "classification_results": results,
        "detailed_reports": reports,
    }
    report_path = os.path.join(run_path, "classification_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nRelatório de classificação salvo em: '{report_path}'")


def print_summary_table(results, input_file_path, feature_type):
    """Imprime a tabela de resumo dos resultados no console."""
    print("\n" + "=" * 60, "\nRELATÓRIO DE COMPARAÇÃO FINAL")
    print(f"Fonte dos Dados: {input_file_path}")
    print(f"Tipo de Feature: {feature_type}")
    print(f"{'Modelo':<25} | {'Acurácia':<10} | {'F1-Score':<10} | {'Tempo (s)':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(
            f"{name:<25} | {metrics['accuracy']:<10.4f} | {metrics['f1_score_weighted']:<10.4f} | {metrics['training_time_seconds']:<10.2f}"
        )
    print("=" * 60)


def main(input_file_path: str):
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 60, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO")
    print(f"Arquivo WSG de entrada: {input_file_path}\n", "=" * 60)

    # --- 1. Gerenciar diretório de saída ---
    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP, run_folder_name="CLASSIFICATION_RUNS"
    )
    run_path = directory_manager.get_run_path()

    # --- 2. Carregar Dados ---
    loader = DirectWSGLoader(file_path=input_file_path)
    wsg_obj = loader.load()

    # --- 3. Executar todos os classificadores ---
    results, reports = run_all_classifiers(wsg_obj, config)

    # --- 4. Salvar e exibir resultados ---
    save_classification_report(run_path, input_file_path, results, reports)
    print_summary_table(results, input_file_path, wsg_obj.metadata.feature_type)

    # --- 5. Finalizar nome do diretório ---
    # Usa a acurácia do GCN como métrica principal para o nome da pasta
    gcn_accuracy = results.get("GCN", {}).get("accuracy", 0.0)
    final_path = directory_manager.finalize_run_directory(
        dataset_name=wsg_obj.metadata.dataset_name, metrics={"gcn_acc": gcn_accuracy}
    )
    print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")


if __name__ == "__main__":
    # O caminho do arquivo é definido aqui
    wsg_file_path = "data/output/EMBEDDING_RUNS/musae-github__loss_0_9483__emb_dim_32__01-10-2025_14-58-11/musae-github_embeddings.wsg.json"

    # E passado para a função main
    main(input_file_path=wsg_file_path)
