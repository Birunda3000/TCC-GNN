import torch
import torch.optim as optim
import os
import random
import numpy as np

# argparse não é mais necessário
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import Config
from src.data_loader import DirectWSGLoader
from src.classifiers import MLPClassifier
from src.train_loop import train_and_evaluate_sklearn_model, run_pytorch_classification


def main(input_file_path):
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 60, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO")
    # CORREÇÃO: Usar a variável recebida pela função
    print(f"Arquivo WSG: {input_file_path}\n", "=" * 60)

    # --- 1. Carregar Dados (Apenas isso!) ---
    # CORREÇÃO: Usar a variável recebida pela função
    loader = DirectWSGLoader(file_path=input_file_path)
    wsg_obj = loader.load()

    results = {}

    # --- 2. Modelos Scikit-learn ---
    print("\n--- Avaliando modelos Scikit-learn ---")
    for name, model in {
        "Regressão Logística": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(random_state=config.RANDOM_SEED),
    }.items():
        acc, f1, train_time = train_and_evaluate_sklearn_model(model, wsg_obj, config)
        results[name] = {
            "Acurácia": acc,
            "F1-Score": f1,
            "Tempo de Treino (s)": train_time,
        }

    # --- 3. Modelo MLP (PyTorch) ---
    print("\n--- Avaliando modelo MLP (PyTorch) ---")
    # A dimensão de entrada e saída é inferida dentro do loop de treino
    mlp_model = MLPClassifier(
        input_dim=(
            config.EMBEDDING_DIM
            if wsg_obj.metadata.feature_type == "sparse_binary"
            else wsg_obj.metadata.num_total_features
        ),
        hidden_dim=128,
        output_dim=len(set(y for y in wsg_obj.graph_structure.y if y is not None)),
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
        "Acurácia": acc,
        "F1-Score": f1,
        "Tempo de Treino (s)": train_time,
    }

    # --- 4. Relatório Final ---
    print("\n" + "=" * 60, "\nRELATÓRIO DE COMPARAÇÃO FINAL")
    # CORREÇÃO: Usar a variável recebida pela função
    print(f"Fonte dos Dados: {input_file_path}")
    print(f"Tipo de Feature: {wsg_obj.metadata.feature_type}")
    print(
        "-" * 60,
        f"\n{'Modelo':<25} | {'Acurácia':<10} | {'F1-Score':<10} | {'Tempo (s)':<10}\n",
        "-" * 60,
    )
    for name, metrics in results.items():
        print(
            f"{name:<25} | {metrics['Acurácia']:<10.4f} | {metrics['F1-Score']:<10.4f} | {metrics['Tempo de Treino (s)']:<10.2f}"
        )
    print("=" * 60, "\n\nRelatório de Classificação Detalhado (MLP):\n", report)


if __name__ == "__main__":
    # O caminho do arquivo é definido aqui
    wsg_file_path = "data/output/EMBEDDING_RUNS/musae-github__loss_0_9483__emb_dim_32__01-10-2025_14-58-11/musae-github_embeddings.wsg.json"

    # E passado para a função main
    main(input_file_path=wsg_file_path)
