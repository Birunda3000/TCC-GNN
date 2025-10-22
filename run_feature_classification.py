import torch
import os
import random
import numpy as np
import json

# --- Importa os modelos base do Scikit-learn ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Importa nossas classes e utilitários customizados ---
from src.config import Config
from src.data_loader import DirectWSGLoader
from src.classifiers import (
    SklearnClassifier, 
    MLPClassifier, 
    XGBoostClassifier,
    TransformerNetworkClassifier
)
from src.directory_manager import DirectoryManager, print_summary_table, save_classification_report

# Verifica se XGBoost está disponível
try:
    import xgboost

    XGBOOST_AVAILABLE = True
except ImportError:
    print("Aviso: XGBoost não está instalado. Executando sem o XGBoostClassifier.")
    XGBOOST_AVAILABLE = False


def main():
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    # Defina aqui o arquivo de embedding que você quer avaliar
    wsg_file_path = "data/output/EMBEDDING_RUNS/musae-github__loss_0_9483__emb_dim_32__01-10-2025_14-58-11/musae-github_embeddings.wsg.json"

    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE EMBEDDINGS")
    print(f"Arquivo de entrada: {wsg_file_path}\n", "=" * 65)

    # --- 2. Carregar Dados e Preparar Ambiente ---
    loader = DirectWSGLoader(file_path=wsg_file_path)
    wsg_obj = loader.load()

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP, run_folder_name="CLASSIFICATION_RUNS"
    )
    run_path = directory_manager.get_run_path()

    results = {}
    reports = {}

    # --- 3. Definir Dimensões e Instanciar Modelos ---
    # A dimensão de entrada é o tamanho do vetor de embedding
    input_dim = len(wsg_obj.node_features["0"].weights)
    # A dimensão de saída é o número de classes únicas
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    # Lista de modelos a serem avaliados
    models_to_run = [
        SklearnClassifier(config, model_class=LogisticRegression, max_iter=1000),
        SklearnClassifier(config, model_class=KNeighborsClassifier, n_neighbors=5),
        SklearnClassifier(config, model_class=RandomForestClassifier),
        MLPClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),
    ]

    # Adiciona XGBoost se disponível
    if XGBOOST_AVAILABLE:
        models_to_run.append(
            XGBoostClassifier(
                config,
                num_boost_round=100,  # Menos rounds para testes rápidos, aumente para melhor performance
                max_depth=6,
                learning_rate=0.1,
            )
        )

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
    save_classification_report(run_path, wsg_file_path, results, reports)
    print_summary_table(results, wsg_file_path, wsg_obj.metadata.feature_type)

    # --- 6. Finalizar Nome do Diretório ---
    # Usa a melhor acurácia como métrica principal para o nome da pasta
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_acc = best_model[1]["accuracy"]
    best_model_name = best_model[0].lower()

    final_path = directory_manager.finalize_run_directory(
        dataset_name=f"{wsg_obj.metadata.dataset_name}_embeddings",
        metrics={"best_acc": best_acc, "model": best_model_name},
    )
    print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")


if __name__ == "__main__":
    main()
